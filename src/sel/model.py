"""Load the base model used by the SEL 27B checkpoint."""

import random
from pathlib import Path

import numpy as np
import torch

from .heads import LinkedPointerHead
from .network import DenseSelectionNetwork

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
CONTEXT = 2048
SEED = 3407
FIXED_THRESHOLDS = {"text": 0.4, "image": 0.3}


def resolve_base(local_path: Path | None) -> Path:
    if local_path is not None:
        path = local_path.expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Expected a directory containing the pinned base snapshot")
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
    )


def load_processor(base: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(base, local_files_only=True)


def build_network(base_path: Path, *, training: bool = False) -> DenseSelectionNetwork:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one CUDA GPU using CUDA_VISIBLE_DEVICES")
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
    if len(targets) != 496:
        raise ValueError(f"Expected 496 text LoRA targets, found {len(targets)}")
    model = get_peft_model(
        base,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets, bias="none"
        ),
    )

    core = model.get_base_model()
    language = core.model.language_model
    language.config.use_cache = False
    full_attention = [
        layer for layer in language.layers if layer.block_type == "full_attention"
    ]
    if len(language.layers) != 64 or len(full_attention) != 16:
        raise ValueError("Expected 16 full-attention and 48 linear-attention blocks")
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
    head = LinkedPointerHead(hidden=language.config.hidden_size, routing=True).to(
        "cuda", torch.float32
    )
    return DenseSelectionNetwork(model, head).train(training)
