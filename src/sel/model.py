"""Load the pinned Qwen backbones used by SEL checkpoints."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .heads import DenseHead, LinkedPointerHead
from .network import DenseSelectionNetwork

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
CONTEXT = 2048
SEED = 3407
FIXED_THRESHOLDS = {"text": 0.4, "image": 0.3}


@dataclass(frozen=True)
class Backbone:
    revision: str
    layers: int
    full_attention: int
    targets: int
    rank: int
    head_kind: str


DENSE_9B = "Qwen/Qwen3.5-9B"
BACKBONES = {
    BASE_MODEL: Backbone(BASE_REVISION, 64, 16, 496, 16, "linked-pointer"),
    DENSE_9B: Backbone(
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 32, 8, 248, 32, "dense"
    ),
}


def resolve_base(local_path: Path | None, *, model_id: str = BASE_MODEL) -> Path:
    if local_path is not None:
        path = local_path.expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Expected a directory containing the pinned base snapshot")
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            revision=BACKBONES[model_id].revision,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
    )


def load_processor(base: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(base, local_files_only=True)


def build_network(
    base_path: Path, *, training: bool = False, model_id: str = BASE_MODEL
) -> DenseSelectionNetwork:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one CUDA GPU using CUDA_VISIBLE_DEVICES")
    spec = BACKBONES[model_id]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    base, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        local_files_only=True,
        output_loading_info=True,
    )
    if info["missing_keys"]:
        raise ValueError(f"Missing base checkpoint weights: {info['missing_keys']}")
    if not base.is_loaded_in_4bit or any(p.is_meta for p in base.parameters()):
        raise ValueError("Expected a fully loaded NF4 backbone")
    targets = [
        name
        for name, module in base.named_modules()
        if name.startswith("model.language_model.layers.")
        and isinstance(module, torch.nn.Linear)
    ]
    if len(targets) != spec.targets:
        raise ValueError(
            f"Expected {spec.targets} text LoRA targets, found {len(targets)}"
        )
    model = get_peft_model(
        base,
        LoraConfig(
            r=spec.rank,
            lora_alpha=2 * spec.rank,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
        ),
    )

    core = model.get_base_model()
    language = core.model.language_model
    language.config.use_cache = False
    full_attention = [
        layer for layer in language.layers if layer.block_type == "full_attention"
    ]
    if (len(language.layers), len(full_attention)) != (
        spec.layers,
        spec.full_attention,
    ):
        raise ValueError(f"Unexpected attention layout for {model_id}")
    for layer in full_attention:
        layer.self_attn.is_causal = False
    if training:
        language.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if any(
        p.requires_grad and ("lora_" not in name or p.dtype != torch.float32)
        for name, p in model.named_parameters()
    ):
        raise ValueError("Expected FP32 LoRA as the only trainable backbone parameters")
    hidden = language.config.hidden_size
    head = (
        DenseHead(hidden=hidden)
        if spec.head_kind == "dense"
        else LinkedPointerHead(hidden=hidden, routing=True)
    )
    network = DenseSelectionNetwork(model, head.to("cuda", torch.float32))
    network.base_model_id = model_id
    return network.train(training)
