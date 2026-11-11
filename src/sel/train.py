"""Train the Run09 recipe and evaluate source-aligned predictions."""

import fcntl
import json
import math
import random
import signal
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import append_metric, atomic_json, checkpoint, restore
from .dataset import load_entry, open_dataset
from .heads import SELECTION_MODES, ordered_runs, predicted_selection_mode
from .metrics import (
    aggregate_scores,
    calibrate_thresholds,
    joint_bce_loss,
    token_learning_rate,
)
from .model import (
    BASE_MODEL,
    BASE_REVISION,
    CONTEXT,
    FIXED_THRESHOLDS,
    SEED,
    build_network,
)
from .network import batch_features

BATCH_SIZE = 4
SHORT_BATCH_LIMIT = 256
PEAK_LR = 3e-5
SAVE_INTERVAL = 50


def objective(network, row, features=None):
    mode = row["selection_mode"]
    if features is None:
        features = network.features(row)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = network.head.dense(*features)
        if mode != "dense":
            pointer_loss = network.head.loss(*features, ordered_runs(row["targets"]))
    targets = row["targets"].to(network.device)
    mask = row["target_mask"].to(network.device)
    loss = joint_bce_loss(dense, targets, mask)[0]
    if mode != "dense":
        loss = pointer_loss + 0.25 * loss
    router = network.head.route_logits(features[1])
    label = torch.tensor([SELECTION_MODES.index(mode)], device=network.device)
    return loss + 0.1 * F.cross_entropy(router[None], label), dense, router, features


def evaluate_split(
    network, processor, directory, entries, *, should_stop=lambda: False
):
    rows, routing = [], Counter()
    network.eval()
    with torch.inference_mode():
        for entry in entries:
            if should_stop():
                return None
            sample = load_entry(entry, directory, processor)
            loss, dense, router, features = objective(network, sample)
            mode = predicted_selection_mode(router, len(features[1]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = dense if mode == "dense" else network.head.predict(*features)
            rows.append(
                {
                    "probabilities": logits.float().sigmoid().cpu(),
                    "targets": sample["targets"],
                    "mask": sample["target_mask"],
                    "loss": float(loss),
                    "modality": entry["media_kind"],
                    "task_id": entry["task_id"],
                }
            )
            routing[sample["selection_mode"], mode] += 1
    return rows, [
        {"expected": expected, "predicted": predicted, "examples": count}
        for (expected, predicted), count in sorted(routing.items())
    ]


def fixed_order(entries):
    groups = [[], []]
    for index, entry in enumerate(entries):
        groups[entry["input_tokens"] > SHORT_BATCH_LIMIT].append(index)
    rng = random.Random(SEED)
    batches, tail = [], []
    for group in groups:
        rng.shuffle(group)
        end = len(group) // BATCH_SIZE * BATCH_SIZE
        batches.extend(group[i : i + BATCH_SIZE] for i in range(0, end, BATCH_SIZE))
        tail.extend(group[end:])
    batches.extend(tail[i : i + BATCH_SIZE] for i in range(0, len(tail), BATCH_SIZE))
    remainder = batches.pop() if len(batches[-1]) < BATCH_SIZE else []
    rng.shuffle(batches)
    return [index for batch in batches for index in batch] + remainder


def backward_batch(network, rows, pad_id):
    if max(row["inputs"]["input_ids"].numel() for row in rows) <= SHORT_BATCH_LIMIT:
        features = batch_features(network, rows, pad_id)
        loss = torch.stack(
            [
                objective(network, row, feature)[0]
                for row, feature in zip(rows, features, strict=True)
            ]
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite batched loss")
        loss.backward()
        return float(loss.detach())

    average = 0.0
    for row in rows:
        loss = objective(network, row)[0] / len(rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite accumulated loss")
        loss.backward()
        average += float(loss.detach())
    return average


def train(
    directory: Path,
    output: Path,
    base: Path,
    processor,
    *,
    resume=False,
    max_steps=None,
):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train(directory, output, base, processor, resume=resume, max_steps=max_steps)


def _train(directory, output, base, processor, *, resume, max_steps):
    latest = output / "latest.pt"
    if latest.is_file() != resume:
        raise ValueError(
            "Use --resume with an existing latest.pt, or choose a new run directory"
        )
    if (output / "completed.json").exists():
        raise ValueError("This run already completed")
    entries, summary = open_dataset(directory)
    config = {
        "recipe": "sel-27b-run09-v1",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset_sha256": summary["manifest_sha256"],
        "train_source_units": summary["split_totals"]["train"]["source_units"],
        "max_input_tokens": CONTEXT,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }
    network = build_network(base, training=True)
    params = [param for param in network.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=PEAK_LR, weight_decay=0.01, fused=True)
    state = {
        "step": 0,
        "tokens": 0,
        "input_tokens": 0,
        "elapsed_seconds": 0.0,
        "thresholds": dict(FIXED_THRESHOLDS),
    }
    sampler = {"cursor": 0, "order": fixed_order(entries["train"])}
    if resume:
        state, sampler = restore(latest, network, optimizer, config)
    atomic_json(output / "config.json", config)
    metrics_path = output / "metrics.jsonl"
    if resume and metrics_path.is_file():
        # A crash can leave log rows beyond the last durable optimizer step.
        lines = []
        for line in metrics_path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                break  # An interrupted append can leave an incomplete last row.
            if json.loads(line)["step"] <= state["step"]:
                lines.append(line)
        replacement = output / ".metrics.tmp"
        replacement.write_text("".join(lines))
        replacement.replace(metrics_path)

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    def save():
        checkpoint(latest, network, optimizer, state, sampler, config)

    def progress(stage):
        atomic_json(output / "progress.json", {"stage": stage, **state})

    started = time.monotonic() - state["elapsed_seconds"]
    pad_id = processor.tokenizer.pad_token_id
    order = sampler["order"]
    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        if not resume:
            save()
        while sampler["cursor"] < len(order) and not stopping:
            if max_steps is not None and state["step"] >= max_steps:
                break
            start = time.monotonic()
            indices = order[sampler["cursor"] : sampler["cursor"] + BATCH_SIZE]
            rows = [
                load_entry(entries["train"][i], directory, processor) for i in indices
            ]
            units = sum(len(row["source_indices"]) for row in rows)
            tokens = sum(row["inputs"]["input_ids"].numel() for row in rows)
            optimizer.zero_grad(set_to_none=True)
            loss = backward_batch(network, rows, pad_id)
            rate = token_learning_rate(
                state["tokens"] + units,
                config["train_source_units"],
                peak=PEAK_LR,
                floor=PEAK_LR / 4,
            )
            optimizer.param_groups[0]["lr"] = rate
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite training gradients")
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - start
            sampler["cursor"] += len(rows)
            state.update(
                step=state["step"] + 1,
                tokens=state["tokens"] + units,
                input_tokens=state["input_tokens"] + tokens,
                epoch=sampler["cursor"] / len(order),
                train_loss=loss,
                lr=rate,
                elapsed_seconds=time.monotonic() - started,
                gradient_norm=norm,
                step_seconds=duration,
                input_tokens_per_second=tokens / duration,
            )
            if state["step"] == 1 or state["step"] % SAVE_INTERVAL == 0:
                save()
            progress("training")
            append_metric(metrics_path, state)
            print(
                f"Step {state['step']} · loss {loss:.4f} · {tokens / duration:,.0f} input tokens/s",
                flush=True,
            )
        save()
        if stopping or sampler["cursor"] < len(order):
            progress("paused")
            return
        if state["input_tokens"] != summary["split_totals"]["train"]["input_tokens"]:
            raise ValueError("Completed input token count differs from the manifest")
        reports = {}
        for split in ("val", "test"):
            progress(f"evaluating_{split}")
            result = evaluate_split(
                network,
                processor,
                directory,
                entries[split],
                should_stop=lambda: stopping,
            )
            if result is None:
                progress("paused")
                return
            rows, routing = result
            if split == "val":
                state["thresholds"] = calibrate_thresholds(rows)
            report = {
                "fixed": aggregate_scores(rows, FIXED_THRESHOLDS),
                "calibrated": aggregate_scores(rows, state["thresholds"]),
                "routing": routing,
            }
            atomic_json(output / f"{split}-metrics.json", report)
            reports[split] = report
        state["elapsed_seconds"] = time.monotonic() - started
        save()
        atomic_json(
            output / "completed.json",
            {"config": config, "state": state, "metrics": reports},
        )
        progress("complete")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
