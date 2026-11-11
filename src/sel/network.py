"""Hidden-state readout from the native multimodal Qwen backbone."""

import torch
from torch.nn.utils.rnn import pad_sequence

from .heads import predicted_selection_mode


class DenseSelectionNetwork(torch.nn.Module):
    """No LM-head forward; the native multimodal model returns hidden states only."""

    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head
        self.core = model.get_base_model()
        self.language = self.core.model.language_model
        self.device = next(model.parameters()).device
        self._last_position_shape = None

    def train(self, mode=True):
        super().train(mode)
        self.core.model.visual.eval()
        return self

    def hidden(self, inputs, *, input_ids_override=None):
        kwargs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k not in {"labels", "offset_mapping"}
        }
        if input_ids_override is not None:
            kwargs["input_ids"] = input_ids_override.to(self.device)
        ids = kwargs["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Expected one complete example per microbatch")
        padding = kwargs.get("attention_mask")
        if isinstance(padding, dict):
            raise ValueError(
                "Official processor inputs must retain their2D padding mask"
            )
        # Native position construction MUST see a Tensor mask, never our bidi mask map.
        if kwargs.get("position_ids") is None:
            kwargs["position_ids"] = self.core.model.compute_3d_position_ids(
                input_ids=ids,
                inputs_embeds=None,
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"),
                attention_mask=padding,
                past_key_values=None,
                mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            )
        position_ids = kwargs.get("position_ids")
        self._last_position_shape = (
            list(position_ids.shape) if position_ids is not None else None
        )
        # Complete examples avoid unpadding; future padded batches retain masks.
        original_padding = inputs.get("attention_mask")
        all_valid = original_padding is None or bool(original_padding.bool().all())
        mask = None if all_valid else padding
        kwargs["attention_mask"] = {"full_attention": mask, "linear_attention": mask}
        kwargs.update(use_cache=False, return_dict=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.core.model(**kwargs).last_hidden_state[0]

    def features(self, example):
        hidden = self.hidden(example["inputs"])
        source = example["source_indices"].to(self.device)
        questions = example["question_positions"].to(self.device)
        return hidden[source], hidden[questions]

    def forward_selection(self, example):
        features = self.features(example)
        mode = example.get("selection_mode")
        if mode is None and getattr(self.head, "router", None) is not None:
            questions = features[1]
            mode = predicted_selection_mode(
                self.head.route_logits(questions), len(questions)
            )
        mode = mode or "dense"
        if example.get("linked_selection", False):
            mode = "linked"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = (
                self.head(*features)
                if mode == "dense"
                else self.head.predict(*features)
            )
        return logits, mode

    def forward(self, example):
        return self.forward_selection(example)[0]


def batch_features(network, examples, pad_id, *, check_positions=False):
    device = network.device
    inputs = [example["inputs"] for example in examples]
    ids = [row["input_ids"][0] for row in inputs]
    pad = pad_sequence(
        [torch.ones_like(x, dtype=torch.bool) for x in ids], batch_first=True
    ).to(device)
    ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device)
    mm_types = pad_sequence(
        [row["mm_token_type_ids"][0] for row in inputs], batch_first=True
    ).to(device)
    images = [row for row in inputs if "pixel_values" in row]
    media = {}
    if images:
        for key in ("pixel_values", "image_grid_thw"):
            media[key] = torch.cat([row[key] for row in images]).to(device)
    core = network.core.model
    positions = core.compute_3d_position_ids(
        input_ids=ids,
        inputs_embeds=None,
        attention_mask=pad,
        mm_token_type_ids=mm_types,
        image_grid_thw=media.get("image_grid_thw"),
    )
    if check_positions:
        assert positions is not None
        for i, row in enumerate(inputs):
            valid_ids = row["input_ids"].to(device)
            single_positions = core.compute_3d_position_ids(
                input_ids=valid_ids,
                inputs_embeds=None,
                attention_mask=row["attention_mask"].to(device),
                mm_token_type_ids=row["mm_token_type_ids"].to(device),
                image_grid_thw=row["image_grid_thw"].to(device)
                if "image_grid_thw" in row
                else None,
            )
            if single_positions is None:
                single_positions = (
                    torch.arange(valid_ids.numel(), device=device)
                    .view(1, 1, -1)
                    .expand(3, 1, -1)
                )
            assert torch.equal(positions[:, i : i + 1, pad[i]], single_positions)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = core(
            input_ids=ids,
            attention_mask={"full_attention": pad, "linear_attention": pad},
            position_ids=positions,
            mm_token_type_ids=mm_types,
            use_cache=False,
            return_dict=True,
            **media,
        ).last_hidden_state
    return [
        (
            hidden[i, row["source_indices"].to(device)],
            hidden[i, row["question_positions"].to(device)],
        )
        for i, row in enumerate(examples)
    ]
