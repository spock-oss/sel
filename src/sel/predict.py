"""Score original text tokens and native image patches."""

from pathlib import Path

import torch

from .data import pack_sources
from .environment import add_environment_prompt, environment_entry
from .model import CONTEXT


def predict(
    network,
    processor,
    *,
    text: str | None,
    image: Path | None,
    queries: list[str],
    thresholds: dict[str, float],
    mode: str | None = None,
    threshold: float | None = None,
    environment: str | None = None,
    query_first: bool = False,
) -> dict:
    if not text and image is None:
        raise ValueError("Provide source text or an image")
    row = pack_sources(
        processor,
        text=text,
        image_path=image,
        queries=queries,
        max_length=CONTEXT,
        query_first=query_first,
    )
    length = row["inputs"]["input_ids"].shape[-1]
    if environment:
        prefix = environment_entry(processor.tokenizer, environment, length)
        length = prefix["input_tokens"]
        if length > CONTEXT:
            raise ValueError(
                f"Input exceeds {CONTEXT} tokens with environment instructions"
            )
        add_environment_prompt(row, prefix)
    if mode:
        row["selection_mode"] = mode

    with torch.inference_mode():
        logits, used_mode = network.forward_selection(row)
        scores = logits.float().sigmoid().tolist()
    units = row["metadata"]["source_units"]
    cutoff = thresholds[row["metadata"]["modality"]] if threshold is None else threshold
    return {
        "mode": used_mode,
        "input_tokens": length,
        "source_units": units,
        "threshold": cutoff if used_mode == "dense" else None,
        "results": [
            {
                "query": query,
                "scores": values,
                "selected_indices": [
                    index
                    for index, score in enumerate(values)
                    if score >= (cutoff if used_mode == "dense" else 0.5)
                ],
            }
            for query, values in zip(queries, scores, strict=True)
        ],
    }
