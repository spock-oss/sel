"""Shared image scoring and replay output for the gameplay examples."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from example import select

CELL = 32
BOARD = "#141d2b"
GRID = "#38485e"
INK = "#f3f6fb"
ACCENT = "#ed826d"
FONT = ImageFont.load_default(size=16)


def parser(description, filename, turns):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--output", type=Path, default=Path(filename), metavar="REPLAY.gif")
    cli.add_argument("--max-turns", type=int, default=turns)
    return cli


def parse_game_args(cli):
    args = cli.parse_args()
    if args.max_turns < 1:
        cli.error("--max-turns must be at least 1")
    if args.output.suffix.lower() != ".gif":
        cli.error("--output must end in .gif")
    return args


def cell_box(cell):
    row, col = cell
    left, top = col * CELL, row * CELL
    return left, top, left + CELL - 1, top + CELL - 1


def choose(url, question, image, candidates):
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    result = select(
        url,
        question=question,
        image_base64=base64.b64encode(encoded.getvalue()).decode("ascii"),
    )
    allowed = set(candidates)
    scores = {}
    for unit in result["units"]:
        if unit["kind"] != "image":
            continue
        left, top, right, bottom = unit["xyxy"]
        cell = (int((top + bottom) / (2 * CELL)), int((left + right) / (2 * CELL)))
        if cell in allowed:
            scores[cell] = max(scores.get(cell, -1.0), unit["score"])
    if missing := allowed - scores.keys():
        raise SystemExit(f"SEL returned no patch for {len(missing)} playable cells.")
    cell = min(scores, key=lambda pos: (-scores[pos], pos))
    return cell, scores[cell]


def replay_frame(image, label, selected=None):
    if selected is not None:
        overlay = Image.new("RGBA", image.size)
        ImageDraw.Draw(overlay).rectangle(cell_box(selected), fill=(230, 110, 92, 105))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        ImageDraw.Draw(image).rectangle(cell_box(selected), outline=ACCENT, width=2)
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
    frame = Image.new("RGB", (max(image.width + 28, 300), image.height + 62), BOARD)
    frame.paste(image, ((frame.width - image.width) // 2, 14))
    ImageDraw.Draw(frame).text((14, image.height + 33), label, font=FONT, fill=INK)
    return frame


def save_replay(output, frames, game, outcome, moves, **setup):
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=[720] * (len(frames) - 1) + [2000],
        disposal=2,
    )
    trace = output.with_suffix(".json")
    trace.write_text(
        json.dumps(dict(game=game, outcome=outcome, moves=moves, **setup), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{outcome} after {len(moves)} moves. Replay: {output}. Move log: {trace}.")
