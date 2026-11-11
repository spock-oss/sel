"""Selection loss, metrics and threshold calibration."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def joint_bce_loss(logits, targets, mask=None, positive_weight=1.0):
    """Choose ONE whole [Q,N] alternative per example, never union/per-question min."""
    if logits.ndim != 2 or targets.ndim != 3 or targets.shape[1:] != logits.shape:
        raise ValueError("Expected logits[Q,N], complete targets[A,Q,N]")
    if targets.shape[0] < 1 or not torch.isfinite(targets).all():
        raise ValueError("At least one finite complete alternative is required")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("BCE targets must be binary")
    mask = torch.ones_like(logits, dtype=torch.bool) if mask is None else mask.bool()
    if mask.shape != logits.shape or not mask.any():
        raise ValueError("Expected a nonempty source-only mask[Q,N]")
    values = F.binary_cross_entropy_with_logits(
        logits.float()[None].expand_as(targets),
        targets.float(),
        reduction="none",
        pos_weight=logits.new_tensor(positive_weight, dtype=torch.float32),
    )
    alternative_losses = (values * mask[None]).sum((1, 2)) / mask.sum()
    loss, selected = alternative_losses.min(0)
    return loss, {
        "alternative_index": int(selected.detach()),
        "alternative_losses": alternative_losses.detach(),
    }


def coherent_scores(probabilities, targets, mask=None, threshold=0.5):
    """All reported questions use the SAME best complete alternative."""
    p = torch.as_tensor(probabilities).float().cpu()
    gold = torch.as_tensor(targets).bool().cpu()
    valid = (
        torch.ones_like(p, dtype=torch.bool)
        if mask is None
        else torch.as_tensor(mask).bool().cpu()
    )
    if (
        p.ndim != 2
        or gold.ndim != 3
        or gold.shape[1:] != p.shape
        or valid.shape != p.shape
    ):
        raise ValueError("Invalid source scoring shapes")
    if not valid.any() or gold.shape[0] == 0:
        raise ValueError("No scored source units or complete alternatives")
    predicted = (p >= threshold) & valid
    gold = gold & valid[None]
    tp = (gold & predicted[None]).sum(-1)
    fp = (~gold & predicted[None] & valid[None]).sum(-1)
    fn = (gold & ~predicted[None]).sum(-1)
    denominator = 2 * tp + fp + fn
    f1 = torch.where(denominator > 0, 2 * tp / denominator, 1.0)
    question_valid = valid.any(-1)
    alternative_means = f1[:, question_valid].mean(-1)
    alternative = int(alternative_means.argmax())
    question_exact = ((gold[alternative] == predicted) | ~valid).all(-1)
    exact_alternatives = ((gold == predicted[None]) | ~valid[None]).all(-1).all(-1)
    return {
        "macro_f1": float(alternative_means[alternative]),
        "joint_exact": bool(exact_alternatives.any()),
        "alternative_index": alternative,
        "questions": [
            {
                "f1": float(f1[alternative, q]),
                "exact": bool(question_exact[q]),
                "tp": int(tp[alternative, q]),
                "fp": int(fp[alternative, q]),
                "fn": int(fn[alternative, q]),
            }
            for q in range(p.shape[0])
            if question_valid[q]
        ],
    }


def aggregate_scores(rows, thresholds):
    by_modality, by_task = {}, {}
    all_questions, joint, losses = [], [], []
    for row in rows:
        modality = row["modality"]
        result = coherent_scores(
            row["probabilities"],
            row["targets"],
            row.get("mask"),
            thresholds.get(modality, 0.5),
        )
        all_questions.extend(result["questions"])
        joint.append(result["joint_exact"])
        if "loss" in row:
            losses.append(row["loss"])
        for key, groups in [(modality, by_modality), (str(row["task_id"]), by_task)]:
            group = groups.setdefault(key, {"questions": [], "joint": [], "loss": []})
            group["questions"].extend(result["questions"])
            group["joint"].append(result["joint_exact"])
            if "loss" in row:
                group["loss"].append(row["loss"])

    def summarize(questions, exacts, group_losses):
        tp, fp, fn = (sum(q[k] for q in questions) for k in ["tp", "fp", "fn"])
        return {
            "macro_f1": float(np.mean([q["f1"] for q in questions]))
            if questions
            else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "question_exact": float(np.mean([q["exact"] for q in questions]))
            if questions
            else 0.0,
            "joint_exact": float(np.mean(exacts)) if exacts else 0.0,
            "questions": len(questions),
            "examples": len(exacts),
            "loss": float(np.mean(group_losses)) if group_losses else None,
        }

    result = summarize(all_questions, joint, losses)
    result["by_modality"] = {
        k: summarize(v["questions"], v["joint"], v["loss"])
        for k, v in by_modality.items()
    }
    result["by_task"] = {
        k: summarize(v["questions"], v["joint"], v["loss"]) for k, v in by_task.items()
    }
    result["task_macro_f1"] = (
        float(np.mean([v["macro_f1"] for v in result["by_task"].values()]))
        if by_task
        else 0.0
    )
    result["thresholds"] = dict(thresholds)
    result["label_policy"] = "One coherent complete alternative per example"
    return result


def calibrate_thresholds(rows):
    """Only callers' validation rows; never call this on held-out test predictions."""
    thresholds = {}
    candidates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for modality in sorted({row["modality"] for row in rows}):
        subset = [row for row in rows if row["modality"] == modality]
        best = max(
            candidates,
            key=lambda t: (
                aggregate_scores(subset, {modality: t})["macro_f1"],
                -abs(t - 0.5),
            ),
        )
        thresholds[modality] = best
    return thresholds


def token_learning_rate(tokens, total_tokens, peak=1e-4, floor=2.5e-5, warmup=0.05):
    if total_tokens <= 0:
        raise ValueError("Training token count must be positive")
    fraction = min(1.0, max(0.0, tokens / total_tokens))
    if fraction < warmup:
        return peak * fraction / warmup
    progress = (fraction - warmup) / (1 - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))
