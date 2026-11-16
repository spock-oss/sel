"""Serve a released dense SEL bundle with FastAPI and vLLM.

Example: python serve.py spock-oss/sel-qwen-0.8b
Use --fp8 for the 9B model on an FP8-capable GPU.
"""

import argparse
import asyncio
import base64
import binascii
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from itertools import count

# The locally registered model and request engine live in this process.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoProcessor
from vllm import LLM, ModelRegistry, PoolingParams
from vllm.inputs.engine import mm_input, tokens_input
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange

BUNDLE_FILES = [
    "sel_config.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "selection_head.safetensors",
]
BASE_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class SelectionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    text: str | None = Field(default=None, max_length=65536)
    image_base64: str | None = Field(
        default=None, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3)
    )
    threshold: float | None = Field(default=None, ge=0, le=1)


class DenseHead(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.source = torch.nn.Linear(hidden, 256)
        self.question = torch.nn.Linear(hidden, 256)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, source, question):
        source = self.source(source)[None]
        question = self.question(question)[:, None]
        joint = source + question + source * question / math.sqrt(source.shape[-1])
        return self.output(F.gelu(joint)).squeeze(-1).float()


class SelectionPooler(Pooler):
    def __init__(self, hidden, head_file):
        super().__init__()
        self.head = DenseHead(hidden)
        self.head.load_state_dict(load_file(head_file))
        self.head.requires_grad_(False)

    def get_supported_tasks(self):
        return {"token_embed"}

    def forward(self, hidden_states, pooling_metadata):
        if pooling_metadata.get_pooling_cursor().is_partial_prefill():
            raise RuntimeError("SEL requires the complete prompt in one prefill")
        outputs = []
        cursor = 0
        for length, params in zip(
            pooling_metadata.prompt_lens.tolist(),
            pooling_metadata.pooling_params,
            strict=True,
        ):
            hidden = hidden_states[cursor : cursor + length]
            cursor += length
            positions = params.extra_kwargs
            if positions is None:  # vLLM startup profiling
                outputs.append(hidden.new_zeros((1, 1)))
                continue
            source = torch.as_tensor(positions["source"], device=hidden.device)
            question = torch.as_tensor(positions["question"], device=hidden.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.head(hidden[source], hidden[question])
            outputs.append(logits.sigmoid().contiguous())
        return outputs


class QwenForSelection(as_embedding_model(Qwen3_5ForConditionalGeneration)):
    def _init_pooler(self, vllm_config, prefix=""):
        config = vllm_config.model_config
        return SelectionPooler(
            config.hf_text_config.hidden_size, config.hf_config.sel_head_file
        )

    def load_weights(self, weights):
        loaded = Qwen3_5ForConditionalGeneration.load_weights(self, weights)
        return loaded | {
            name for name, _ in self.named_parameters() if name.startswith("pooler.")
        }


def decode_image(value):
    try:
        data = base64.b64decode(value, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Images must be at most 8 MiB")
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Images must be at most 16 megapixels")
            return image.convert("RGB")
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Provide a valid base64-encoded image") from exc


def pack_sources(processor, question, text, image, limit):
    """Use the question-first format of the released dense checkpoints."""
    tokenizer = processor.tokenizer
    if any(
        token in question or token in text for token in tokenizer.all_special_tokens
    ):
        raise ValueError("The question and text must contain no model control tokens")
    question_line = "Question 1: " + question
    ending = "\nEnd selection questions.\n" if question != question.rstrip() else "\n"
    header = "Selection questions:\n"
    content = [{"type": "text", "text": "\n" + header + question_line + ending}]
    if image is not None:
        content.extend(
            [
                {
                    "type": "text",
                    "text": "Source image (rules, objects and selectable surface):\n",
                },
                {"type": "image"},
            ]
        )
    marker = "\nSource text:\n"
    if text:
        content.append({"type": "text", "text": marker + text + "\nEnd source text.\n"})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_args = {}
    if image is not None:
        image_args = {
            "images": [image],
            "images_kwargs": {
                "size": {"shortest_edge": 65536, "longest_edge": 6000 * 32 * 32}
            },
        }
    encoded = processor(
        text=prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
        **image_args,
    )
    inputs = {
        key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > limit:
        raise ValueError(f"Input has {len(ids)} tokens. The limit is {limit}.")
    expanded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    mapped = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if mapped["input_ids"] != ids:
        raise ValueError("Could not align the source with the expanded token sequence")
    offsets = mapped["offset_mapping"]
    question_start = expanded.index(header) + len(header)
    question_end = question_start + len(question_line)
    if expanded[question_start:question_end] != question_line:
        raise ValueError("Could not align the question with the token sequence")
    question_positions = [
        pos
        for pos, (left, right) in enumerate(offsets)
        if left < question_end and right > question_start
    ]
    if not question_positions:
        raise ValueError("The question contains no tokens")

    source = []
    if image is not None:
        patches = [
            i for i, token in enumerate(ids) if token == processor.image_token_id
        ]
        frames, height, width = map(int, inputs["image_grid_thw"][0])
        merge = processor.image_processor.merge_size
        rows, cols = height // merge, width // merge
        if frames != 1 or len(patches) != rows * cols or not patches:
            raise ValueError("Could not align the image grid with the token sequence")
        for idx, pos in enumerate(patches):
            row, col = divmod(idx, cols)
            source.append(
                (
                    pos,
                    {
                        "kind": "image",
                        "xyxy": [
                            col * image.width / cols,
                            row * image.height / rows,
                            (col + 1) * image.width / cols,
                            (row + 1) * image.height / rows,
                        ],
                        "grid": [row, col, rows, cols],
                    },
                )
            )
    if text:
        start = expanded.index(marker, question_end) + len(marker)
        end = start + len(text)
        if expanded[start:end] != text:
            raise ValueError("Could not align the text with the token sequence")
        for pos, (left, right) in enumerate(offsets):
            if left < end and right > start:
                source.append(
                    (
                        pos,
                        {
                            "kind": "text",
                            "start": max(left - start, 0),
                            "end": min(right - start, len(text)),
                        },
                    )
                )
    if not source:
        raise ValueError("The request contains no selectable source positions")
    source.sort(key=lambda item: item[0])
    indices, units = zip(*source, strict=True)
    return inputs, list(indices), [question_positions[-1]], units


class Selector:
    def __init__(self, repo, revision, fp8):
        from pathlib import Path

        bundle = Path(
            snapshot_download(repo, revision=revision, allow_patterns=BUNDLE_FILES)
        )
        config = json.loads((bundle / "sel_config.json").read_text())
        base_id = config["base_model"]
        base_revision = config["base_revision"]
        if (
            config.get("format_version") != 1
            or config.get("head_kind") != "dense"
            or config.get("query_first") is not True
            or config.get("max_questions") != 1
            or BASE_REVISIONS.get(base_id) != base_revision
        ):
            raise ValueError(
                "Expected a released question-first SEL Qwen 0.8B or 9B bundle"
            )
        self.limit = config["max_input_tokens"]
        self.thresholds = config["fixed_thresholds"]
        adapter = json.loads((bundle / "adapter_config.json").read_text())
        base = snapshot_download(
            base_id,
            revision=base_revision,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        self.ids = count()
        arch = "QwenForSelection"
        ModelRegistry.register_model(arch, QwenForSelection)
        quantization = {}
        if fp8:
            quantization = {
                "quantization": "fp8_per_channel",
                "quantization_config": {"ignore": [r"re:.*visual\..*"]},
            }
        self.llm = LLM(
            model=base,
            tokenizer=base,
            dtype="bfloat16",
            runner="pooling",
            convert="none",
            pooler_config={"task": "token_embed"},
            hf_overrides={
                "architectures": [arch],
                "is_causal": False,
                "text_config": {"is_causal": False},
                "sel_head_file": str(bundle / "selection_head.safetensors"),
            },
            max_model_len=self.limit,
            max_num_batched_tokens=self.limit,
            max_num_seqs=1,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            attention_backend="FLASH_ATTN",
            compilation_config={"cudagraph_mode": "NONE"},
            kv_cache_memory_bytes=512 << 20,
            gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_cache_gb=0,
            mm_device_do_normalize=False,
            skip_mm_profiling=True,
            enable_lora=True,
            max_lora_rank=adapter["r"],
            lora_target_modules=sorted(
                {name.rsplit(".", 1)[-1] for name in adapter["target_modules"]}
            ),
            **quantization,
        )
        self.engine = self.llm.llm_engine
        self.adapter = LoRARequest("sel", 1, str(bundle))
        if not self.engine.add_lora(self.adapter):
            raise RuntimeError("vLLM could not load the SEL LoRA")
        hf = self.engine.model_config.hf_config
        self.fields = _create_qwen2vl_field_factory(hf.vision_config.spatial_merge_size)

    def select(self, request):
        text = request.text or ""
        image = decode_image(request.image_base64) if request.image_base64 else None
        if not request.question.strip() or (not text.strip() and image is None):
            raise ValueError("Provide a question and at least one source")
        inputs, source, question, units = pack_sources(
            self.processor, request.question, text, image, self.limit
        )
        ids = inputs["input_ids"][0].tolist()
        identifier = str(next(self.ids))
        prompt = tokens_input(ids)
        if image is not None:
            patches = [
                i
                for i, token in enumerate(ids)
                if token == self.processor.image_token_id
            ]
            if patches[-1] - patches[0] + 1 != len(patches):
                raise ValueError("Expected one contiguous image token range")
            prompt = mm_input(
                ids,
                MultiModalKwargsItems.from_hf_inputs(inputs, self.fields(inputs)),
                {"image": [f"image:{identifier}"]},
                {"image": [PlaceholderRange(patches[0], len(patches))]},
            )
        params = PoolingParams(
            task="token_embed", extra_kwargs={"source": source, "question": question}
        )
        try:
            self.engine.add_request(
                identifier, prompt, params, lora_request=self.adapter
            )
            while self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    if output.finished:
                        scores = output.outputs.data[0].tolist()
                        break
                else:
                    continue
                break
            else:
                raise RuntimeError("vLLM completed without selection scores")
        except Exception:
            self.engine.abort_request([identifier])
            raise
        modality = ("mixed" if text else "image") if image is not None else "text"
        threshold = (
            self.thresholds[modality]
            if request.threshold is None
            else request.threshold
        )
        return {
            "input_tokens": len(ids),
            "threshold": threshold,
            "units": [
                {**unit, "score": score, "selected": score >= threshold}
                for unit, score in zip(units, scores, strict=True)
            ],
        }

    def close(self):
        self.engine.engine_core.shutdown()


def create_app(repo, revision, fp8):
    @asynccontextmanager
    async def lifespan(app):
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            selector = await loop.run_in_executor(
                executor, Selector, repo, revision, fp8
            )
            app.state.selector = selector
            app.state.executor = executor
            try:
                yield
            finally:
                await loop.run_in_executor(executor, selector.close)

    app = FastAPI(title="SEL", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "max_input_tokens": app.state.selector.limit}

    @app.post("/select")
    async def select(request: SelectionRequest):
        try:
            return await asyncio.get_running_loop().run_in_executor(
                app.state.executor, app.state.selector.select, request
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face model ID or model page URL")
    parser.add_argument("--revision", help="Optional SEL bundle commit or tag")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--fp8", action="store_true", help="Use FP8 for the language backbone"
    )
    args = parser.parse_args()
    repo = args.repo.removeprefix("https://huggingface.co/").rstrip("/")
    uvicorn.run(
        create_app(repo, args.revision, args.fp8), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
