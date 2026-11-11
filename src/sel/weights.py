"""Package adapter and selection-head tensors independently of optimizer state."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .checkpoint import atomic_json
from .model import BASE_MODEL, BASE_REVISION, CONTEXT, FIXED_THRESHOLDS

FORMAT_VERSION = 1
LORA_PREFIX = "base_model.model."
LORA_SUFFIX = ".lora_A.weight"


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    # Optimizer and RNG state in a local training checkpoint require trusted pickle.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    adapter, head = state["adapter"], state["head"]
    targets = sorted(
        key.removeprefix(LORA_PREFIX).removesuffix(LORA_SUFFIX)
        for key in adapter
        if key.startswith(LORA_PREFIX) and key.endswith(LORA_SUFFIX)
    )
    if (
        len(targets) != 496
        or len(adapter) != 992
        or not any(key.startswith("router.") for key in head)
    ):
        raise ValueError(
            "Expected a SEL 27B checkpoint with LoRA and a routed selection head"
        )
    if any(
        value.shape[0] != 16
        for key, value in adapter.items()
        if key.endswith(LORA_SUFFIX)
    ):
        raise ValueError("Expected rank-16 LoRA weights")
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
                "base_model_name_or_path": BASE_MODEL,
                "revision": BASE_REVISION,
                "r": 16,
                "lora_alpha": 32,
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
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "max_input_tokens": CONTEXT,
                "step": state["state"]["step"],
                "fixed_thresholds": FIXED_THRESHOLDS,
                "calibrated_thresholds": state["state"]["thresholds"],
            },
        )
        bundle.rename(output)


def load_weights(network, directory: Path) -> dict:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    config = json.loads((directory / "sel_config.json").read_text())
    expected_identity = (FORMAT_VERSION, BASE_MODEL, BASE_REVISION, CONTEXT)
    actual_identity = tuple(
        config[key]
        for key in ("format_version", "base_model", "base_revision", "max_input_tokens")
    )
    if actual_identity != expected_identity:
        raise ValueError(f"Incompatible SEL bundle: {actual_identity}")
    adapter = load_file(directory / "adapter_model.safetensors")
    expected = get_peft_model_state_dict(network.model)
    if adapter.keys() != expected.keys() or any(
        tensor.shape != expected[key].shape for key, tensor in adapter.items()
    ):
        raise ValueError("Adapter tensors do not match the SEL 27B backbone")
    result = set_peft_model_state_dict(network.model, adapter)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
    network.head.load_state_dict(
        load_file(directory / "selection_head.safetensors"), strict=True
    )
    return config
