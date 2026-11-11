"""Prepare and read the three frozen dataset splits."""

import hashlib
import json
import tempfile
from pathlib import Path

from .checkpoint import atomic_json, manifest_fingerprint, read_manifests
from .data import load_cached_example, load_example, save_cached_example
from .environment import add_environment_prompt, environment_entry
from .heads import SELECTION_MODES
from .model import BASE_MODEL, BASE_REVISION, CONTEXT

SPLITS = ("train", "val", "test")


def cache_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_entry(entry: dict, data_dir: Path, processor) -> dict:
    path = data_dir / entry["cache_path"]
    digest = entry.get("cache_sha256")
    if digest is not None and cache_digest(path) != digest:
        raise ValueError(f"Packed record changed since preparation: {path}")
    sample = load_cached_example(path, processor)
    if "environment_ids" in entry:
        add_environment_prompt(sample, entry)
    sample["selection_mode"] = entry["selection_mode"]
    length = sample["inputs"]["input_ids"].shape[-1]
    source, questions = sample["source_indices"], sample["question_positions"]
    if length != entry["input_tokens"] or length > CONTEXT:
        raise ValueError(
            "Packed example length differs from the manifest or exceeds 2,048"
        )
    if not source.numel() or not questions.numel():
        raise ValueError("Each example must contain a source and a question")
    if (
        min(int(source.min()), int(questions.min())) < 0
        or max(int(source.max()), int(questions.max())) >= length
    ):
        raise ValueError("Invalid source or question positions")
    if sample["targets"].shape[1:] != (len(questions), len(source)):
        raise ValueError("Target shape differs from source and question positions")
    return sample


def open_dataset(directory: Path) -> tuple[dict, dict]:
    entries = read_manifests(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if summary["manifest_sha256"] != manifest_fingerprint(directory):
        raise ValueError("Dataset manifests have changed since preparation")
    if summary["context_limit"] != CONTEXT:
        raise ValueError(f"Expected a dataset prepared at {CONTEXT} input tokens")
    if summary.get("base_revision", BASE_REVISION) != BASE_REVISION:
        raise ValueError(
            "Dataset was prepared using a different base tokenizer revision"
        )
    return entries, summary


def prepare(source: Path, output: Path, processor) -> dict:
    """Encode local raw records and preserve the supplied splits."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_records = set()
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        packed = Path(temp) / "packed"
        packed.mkdir()
        totals = {}
        for split in SPLITS:
            entries = [
                json.loads(line)
                for line in (source / f"{split}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if not entries:
                raise ValueError(f"Empty dataset split: {split}")
            totals[split] = {
                "examples": len(entries),
                "input_tokens": 0,
                "source_units": 0,
            }
            with (packed / f"{split}.jsonl").open("w") as manifest:
                for index, entry in enumerate(entries):
                    record = (source / entry["artifact_path"]).resolve(strict=True)
                    if record in seen_records:
                        raise ValueError(f"Record appears more than once: {record}")
                    seen_records.add(record)
                    mode = entry.get("selection_mode", "dense")
                    if mode not in SELECTION_MODES:
                        raise ValueError(f"Unknown selection mode: {mode}")
                    sample = load_example(
                        record,
                        processor,
                        CONTEXT,
                        query_first=entry.get("query_first", False),
                    )
                    length = sample["inputs"]["input_ids"].shape[-1]
                    kind = sample["metadata"]["modality"]
                    if kind not in ("text", "image"):
                        raise ValueError(f"Unsupported training modality: {kind}")
                    if mode == "linked" and len(sample["question_positions"]) < 2:
                        raise ValueError(
                            "Linked selection needs at least two questions"
                        )
                    if mode != "dense" and not sample["targets"].any(dim=-1).all():
                        raise ValueError(
                            "Pointer modes need at least one target per question and alternative"
                        )
                    row = {
                        "cache_path": f"cache/{split}/{index:06d}.pt.gz",
                        "task_id": entry.get("task_id", sample["metadata"]["task_id"]),
                        "selection_mode": mode,
                        "input_tokens": length,
                        "source_units": len(sample["source_indices"]),
                        "media_kind": kind,
                    }
                    if entry.get("environment_prompt"):
                        row.update(
                            environment_entry(
                                processor.tokenizer, entry["environment_prompt"], length
                            )
                        )
                    if row["input_tokens"] > CONTEXT:
                        raise ValueError(
                            f"{record} exceeds {CONTEXT} tokens with environment instructions"
                        )
                    cache_path = packed / row["cache_path"]
                    save_cached_example(sample, cache_path)
                    row["cache_sha256"] = cache_digest(cache_path)
                    totals[split]["input_tokens"] += row["input_tokens"]
                    totals[split]["source_units"] += row["source_units"]
                    manifest.write(json.dumps(row) + "\n")
        summary = {
            "stage": "ready",
            "context_limit": CONTEXT,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "manifest_sha256": manifest_fingerprint(packed),
            "split_totals": totals,
        }
        atomic_json(packed / "summary.json", summary)
        packed.rename(output)
    return summary
