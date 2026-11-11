"""Dense scoring and recurrent pointers over source positions."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class DenseHead(torch.nn.Module):
    def __init__(self, hidden=2560, width=256):
        super().__init__()
        self.source = torch.nn.Linear(hidden, width)
        self.question = torch.nn.Linear(hidden, width)
        self.output = torch.nn.Linear(width, 1)

    def forward(self, source, questions):
        s, q = self.source(source)[None], self.question(questions)[:, None]
        return (
            self.output(F.gelu(s + q + s * q / math.sqrt(s.shape[-1])))
            .squeeze(-1)
            .float()
        )


MAX_SOURCE_RUNS = 32
SELECTION_MODES = ("dense", "one_coherent_answer", "linked")


def predicted_selection_mode(logits, n_questions):
    valid = (0, 1) if n_questions == 1 else (0, 2)
    return SELECTION_MODES[valid[int(logits[list(valid)].argmax())]]


def ordered_runs(targets):
    """Serialize each complete alternative as maximal inclusive source runs."""
    alternatives = set()
    for alternative in targets.bool().tolist():
        grouped = []
        for row in alternative:
            runs = []
            start = None
            for i, bit in enumerate([*row, False]):
                if bit and start is None:
                    start = i
                elif not bit and start is not None:
                    runs.append((start, i - 1))
                    start = None
            grouped.append(tuple(runs))
        alternatives.add(tuple(grouped))
    return sorted(alternatives)


class LinkedPointerHead(nn.Module):
    """One backbone pass. A small head emits ordered source runs and STOP."""

    def __init__(self, hidden=2560, width=256, max_runs=MAX_SOURCE_RUNS, routing=False):
        super().__init__()
        self.base = DenseHead(hidden, width)
        self.position = nn.Linear(1, width, bias=False)
        self.action_type = nn.Embedding(2, width)
        self.stop = nn.Parameter(torch.zeros(width))
        self.cell = nn.GRU(width, width, batch_first=True)
        self.keys = nn.Linear(width, width)
        self.readout = nn.Linear(width, width)
        self.stop_score = nn.Linear(width, 1)
        self.max_runs = max_runs
        self.width = width
        self.router = nn.Linear(hidden, len(SELECTION_MODES)) if routing else None

    def route_logits(self, question_features):
        with torch.autocast(device_type=question_features.device.type, enabled=False):
            return self.router(question_features.float().mean(dim=0))

    def project(self, source, questions):
        source = self.base.source(source)
        questions = self.base.question(questions)
        axis = torch.linspace(
            0, 1, len(source), device=source.device, dtype=source.dtype
        )
        keys = self.keys(source + self.position(axis[:, None]))
        scale = math.sqrt(self.width)
        prior = self.base.output(
            F.gelu(
                source[None]
                + questions[:, None]
                + source[None] * questions[:, None] / scale
            )
        ).squeeze(-1)
        return source, questions, keys, prior

    def dense(self, source, questions):
        return self.base(source, questions)

    def forward(self, source, questions):
        return self.dense(source, questions)

    def scores(self, decoded, keys, prior, question, kind, lo):
        n = keys.shape[0]
        scores = (self.readout(decoded) @ keys.T / math.sqrt(self.width)).float()
        scores = scores + prior[question].float()
        keep = torch.arange(n, device=keys.device) >= lo[..., None]
        scores = scores.masked_fill(~keep, -torch.inf)
        stop = (
            self.stop_score(decoded)
            .float()
            .masked_fill(kind[..., None] != 0, -torch.inf)
        )
        return torch.cat((scores, stop), dim=-1)

    def loss(self, source, questions, alternatives):
        source, questions, keys, prior = self.project(source, questions)
        n = len(source)
        trajectories = []
        for alternative in alternatives:
            previous = n
            steps = []
            for question, spans in enumerate(alternative):
                lo = 0
                for start, end in spans:
                    steps.append((previous, question, 0, lo, start))
                    steps.append((start, question, 1, start, end))
                    previous, lo = end, end + 2
                steps.append((previous, question, 0, lo, n))
                previous = n
            trajectories.append(steps)
        longest = max(map(len, trajectories))
        # A padding decision can safely emit STOP and contributes no likelihood.
        padded = [
            row + [(n, 0, 0, n, n)] * (longest - len(row)) for row in trajectories
        ]
        device = source.device
        previous, question, kind, lo, target = torch.tensor(
            padded, device=device
        ).unbind(-1)
        memory = torch.cat((source, self.stop[None].to(source.dtype)))
        hidden, _ = self.cell(
            questions[question] + memory[previous] + self.action_type(kind)
        )
        scores = self.scores(hidden, keys, prior, question, kind, lo)
        selected = scores.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
        lengths = torch.tensor([len(row) for row in trajectories], device=device)
        loss = -selected.sum(-1).logsumexp(0) / lengths.float().mean()
        return loss

    def predict(
        self,
        source,
        questions,
        *,
        first_runs=None,
        reset_between=False,
        points_only=False,
    ):
        source, questions, keys, prior = self.project(source, questions)
        n = len(source)
        device = source.device
        previous = self.stop.to(source.dtype)
        hidden = None
        predictions = torch.zeros_like(prior, dtype=torch.bool)

        def decision(query, kind, lo, forced=None, require_position=False):
            nonlocal hidden
            x = questions[query] + previous + self.action_type.weight[kind]
            decoded, hidden = self.cell(x[None, None], hidden)
            if forced is not None:
                return forced
            scores = self.scores(
                decoded[0, 0],
                keys,
                prior,
                query,
                torch.tensor(kind, device=device),
                torch.tensor(lo, device=device),
            )
            if require_position:
                scores = scores[:-1]
            return int(scores.argmax())

        for q in range(len(questions)):
            lo = 0
            if points_only:
                start = decision(q, 0, lo, require_position=True)
                previous = source[start]
                decision(q, 1, start, forced=start)
                predictions[q, start] = True
                decision(q, 0, start + 2, forced=n)
                previous = self.stop.to(source.dtype)
                continue
            fixed = first_runs if q == 0 else None
            for i in range(self.max_runs):
                candidate = fixed[i] if fixed is not None and i < len(fixed) else None
                forced_start = (
                    None if fixed is None else n if candidate is None else candidate[0]
                )
                start = decision(q, 0, lo, forced_start)
                if start == n:
                    break
                previous = source[start]
                end = decision(
                    q, 1, start, candidate[1] if candidate is not None else None
                )
                predictions[q, start : end + 1] = True
                previous, lo = source[end], end + 2
            previous = self.stop.to(source.dtype)
            if reset_between:
                hidden = None
        return torch.where(predictions, 8.0, -8.0)
