"""Local resumable checkpoints and atomic writes."""

import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(value, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint(path, network, optimizer, state, sampler, config):
    from peft import get_peft_model_state_dict

    payload = {
        "format_version": 1,
        "adapter": get_peft_model_state_dict(network.model),
        "head": network.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "sampler": sampler,
        "config": config,
        "scheduler": {
            "kind": "source-token warmup cosine",
            "tokens": state["tokens"],
            "total_tokens": config["train_source_units"],
        },
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def restore(path, network, optimizer, config):
    from peft import set_peft_model_state_dict

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != config:
        raise ValueError(
            "Checkpoint configuration/corpus changed; exact resume refused"
        )
    set_peft_model_state_dict(network.model, saved["adapter"])
    network.head.load_state_dict(saved["head"])
    optimizer.load_state_dict(saved["optimizer"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["state"], saved["sampler"]


def append_metric(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


def manifest_fingerprint(data_dir):
    h = hashlib.sha256()
    for split in ["train", "val", "test"]:
        path = Path(data_dir) / f"{split}.jsonl"
        h.update(split.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def read_manifests(data_dir):
    summary_path = Path(data_dir) / "summary.json"
    if (
        not summary_path.exists()
        or json.loads(summary_path.read_text()).get("stage") != "ready"
    ):
        raise ValueError(
            "CPU dataset preparation is incomplete; require immutable ready summary"
        )
    result = {}
    for split in ["train", "val", "test"]:
        path = Path(data_dir) / f"{split}.jsonl"
        result[split] = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if not result["train"] or not result["val"] or not result["test"]:
        raise ValueError("Need nonempty frozen train/val/test manifests")
    return result
