"""Package adapter and selection-head tensors independently of optimizer state."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .checkpoint import atomic_json
from .model import BACKBONES, BASE_MODEL, CONTEXT, FIXED_THRESHOLDS

FORMAT_VERSION = 1
LORA_PREFIX = "base_model.model."
LORA_SUFFIX = ".lora_A.weight"


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    # Optimizer and RNG state in a local training checkpoint require trusted pickle.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    adapter, head = state["adapter"], state["head"]
    model_id = state.get("config", {}).get("model_repo", BASE_MODEL)
    spec = BACKBONES[model_id]
    dense = spec.head_kind == "dense"
    targets = sorted(
        key.removeprefix(LORA_PREFIX).removesuffix(LORA_SUFFIX)
        for key in adapter
        if key.startswith(LORA_PREFIX) and key.endswith(LORA_SUFFIX)
    )
    if (
        len(targets) != spec.targets
        or len(adapter) != 2 * spec.targets
        or any(key.startswith("router.") for key in head) == dense
    ):
        raise ValueError(
            f"Unexpected LoRA targets or {spec.head_kind} head for {model_id}"
        )
    if any(
        value.shape[0] != spec.rank
        for key, value in adapter.items()
        if key.endswith(LORA_SUFFIX)
    ):
        raise ValueError(f"Expected rank-{spec.rank} LoRA weights")
    thresholds = {**FIXED_THRESHOLDS, "mixed": 0.3} if dense else FIXED_THRESHOLDS
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        save_file(adapter, bundle / "adapter_model.safetensors")
        save_file(head, bundle / "selection_head.safetensors")
        atomic_json(
            bundle / "adapter_config.json",
            {
                "peft_type": "LORA",
                "base_model_name_or_path": model_id,
                "revision": spec.revision,
                "r": spec.rank,
                "lora_alpha": 2 * spec.rank,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": targets,
                "inference_mode": True,
            },
        )
        atomic_json(
            bundle / "sel_config.json",
            {
                "format_version": FORMAT_VERSION,
                "base_model": model_id,
                "base_revision": spec.revision,
                "max_input_tokens": CONTEXT,
                "step": state["state"]["step"],
                "fixed_thresholds": thresholds,
                "calibrated_thresholds": state["state"].get("thresholds"),
                "head_kind": spec.head_kind,
                "query_first": dense,
                "max_questions": 1 if dense else None,
            },
        )
        bundle.rename(output)


def read_config(directory: Path) -> dict:
    config = json.loads((directory / "sel_config.json").read_text())
    spec = BACKBONES[config["base_model"]]
    expected_identity = (FORMAT_VERSION, config["base_model"], spec.revision, CONTEXT)
    actual_identity = tuple(
        config[key]
        for key in ("format_version", "base_model", "base_revision", "max_input_tokens")
    )
    if actual_identity != expected_identity:
        raise ValueError(f"Incompatible SEL bundle: {actual_identity}")
    if config.get("head_kind", "linked-pointer") != spec.head_kind:
        raise ValueError("Selection head differs from the pinned model configuration")
    if spec.head_kind == "dense" and (
        config.get("query_first") is not True or config.get("max_questions") != 1
    ):
        raise ValueError("A dense SEL bundle requires one question before the source")
    return config


def load_weights(network, directory: Path) -> dict:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    config = read_config(directory)
    if config["base_model"] != network.base_model_id:
        raise ValueError("The SEL bundle and loaded backbone differ")
    adapter = load_file(directory / "adapter_model.safetensors")
    expected = get_peft_model_state_dict(network.model)
    if adapter.keys() != expected.keys() or any(
        tensor.shape != expected[key].shape for key, tensor in adapter.items()
    ):
        raise ValueError("Adapter tensors do not match the loaded SEL backbone")
    result = set_peft_model_state_dict(network.model, adapter)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
    network.head.load_state_dict(
        load_file(directory / "selection_head.safetensors"), strict=True
    )
    return config
