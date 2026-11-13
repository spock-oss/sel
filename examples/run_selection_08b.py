"""Run the published SEL Qwen 0.8B adapter and selection head."""

import argparse
import json
from pathlib import Path

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file
from sel.heads import SELECTION_MODES, LinkedPointerHead
from sel.network import DenseSelectionNetwork
from sel.predict import predict
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5 import modeling_qwen3_5


def load_model(weights, config, base):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU through CUDA_VISIBLE_DEVICES")
    if base is None:
        base = Path(
            snapshot_download(
                config["base_model"],
                revision=config["base_revision"],
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            )
        )
    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        base,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        local_files_only=True,
    )
    language = backbone.model.language_model
    language.config.use_cache = False
    for layer in language.layers:
        if layer.block_type == "full_attention":
            layer.self_attn.is_causal = False

    model = PeftModel.from_pretrained(backbone, weights)
    head = LinkedPointerHead(hidden=language.config.hidden_size, routing=True).to(
        device="cuda", dtype=torch.float32
    )
    head.load_state_dict(load_file(weights / "selection_head.safetensors"), strict=True)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    return DenseSelectionNetwork(model, head).eval(), processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--base", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--query", required=True, action="append")
    parser.add_argument("--environment")
    parser.add_argument("--mode", choices=SELECTION_MODES)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--query-first", action="store_true")
    args = parser.parse_args()
    text = args.text_file.read_text() if args.text_file else args.text
    if not text and args.image is None:
        parser.error("Provide source text or an image")

    config = json.loads((args.weights / "sel_config.json").read_text())
    network, processor = load_model(args.weights, config, args.base)
    result = predict(
        network,
        processor,
        text=text,
        image=args.image,
        queries=args.query,
        thresholds=config["calibrated_thresholds"],
        mode=args.mode,
        threshold=args.threshold,
        environment=args.environment,
        query_first=args.query_first,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
