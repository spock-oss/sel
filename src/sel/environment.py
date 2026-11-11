"""Attach input-only environment text to an unpadded native SEL example."""

import torch

CHAT_HEADER = "<|im_start|>user\n"
# Qwen user role marker and newline. Preparation verifies native encoding.
CHAT_HEADER_IDS = (248045, 846, 198)


def environment_entry(tokenizer, text, original_length):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Environment instructions must be nonempty text")
    ids = tokenizer.encode(
        f"Environment and selection rules:\n{text}\n\n", add_special_tokens=False
    )
    if set(ids).intersection(tokenizer.all_special_ids):
        raise ValueError("Environment instructions may not include control tokens")
    return {
        "environment_prompt": text,
        "environment_ids": ids,
        "input_tokens": original_length + len(ids),
    }


def add_environment_prompt(example, entry):
    """Add a chat prefix while leaving the complete source and its labels intact."""
    inputs = example["inputs"]
    ids = inputs["input_ids"]
    prefix_length = len(CHAT_HEADER_IDS)
    metadata = example["metadata"]
    if ids[0, :prefix_length].tolist() != list(CHAT_HEADER_IDS):
        raise ValueError("Unexpected native chat header")
    extra = entry["environment_ids"]
    for key, value in (
        ("input_ids", extra),
        ("attention_mask", [1] * len(extra)),
        ("mm_token_type_ids", [0] * len(extra)),
    ):
        original = inputs[key]
        if original.shape != ids.shape:
            raise ValueError(f"Unexpected sequence shape for {key}")
        prefix = original.new_tensor([value])
        inputs[key] = torch.cat(
            (original[:, :prefix_length], prefix, original[:, prefix_length:]), dim=1
        )
    if inputs["input_ids"].shape[-1] != entry["input_tokens"]:
        raise ValueError("Prompt prefix does not match the manifest's token count")
    if "position_ids" in inputs:
        raise ValueError("Native positions must be rebuilt after adding instructions")
    for key in ("source_indices", "question_positions"):
        if (example[key] < prefix_length).any():
            raise ValueError(f"Cannot move a position in the chat header: {key}")
        example[key] = example[key] + len(extra)
    card = entry["environment_prompt"]
    prompt_prefix = f"Environment and selection rules:\n{card}\n\n"
    for key in ("prompt", "expanded_prompt"):
        previous = metadata[key]
        if not previous.startswith(CHAT_HEADER):
            raise ValueError(f"Unexpected chat header in {key}")
        metadata[key] = CHAT_HEADER + prompt_prefix + previous[len(CHAT_HEADER) :]
    metadata["environment_prompt"] = card
    metadata["sequence_length"] = entry["input_tokens"]
    return example
