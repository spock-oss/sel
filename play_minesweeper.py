"""Play Minesweeper one click at a time using a running SEL server."""

import random

from PIL import Image, ImageDraw

from _gameplay import (
    BOARD,
    CELL,
    FONT,
    GRID,
    INK,
    cell_box,
    choose,
    parse_game_args,
    parser,
    replay_frame,
    save_replay,
)

SIZE = 8
START = (3, 3)
MINE_COUNT = 10
CELLS = [(row, col) for row in range(SIZE) for col in range(SIZE)]


def neighbors(cell):
    row, col = cell
    return [
        (r, c)
        for r in range(max(0, row - 1), min(SIZE, row + 2))
        for c in range(max(0, col - 1), min(SIZE, col + 2))
        if (r, c) != cell
    ]


def reveal(cell, clues, revealed):
    pending = [cell]
    while pending:
        pos = pending.pop()
        if pos in revealed:
            continue
        revealed.add(pos)
        if clues[pos] == 0:
            pending.extend(neighbors(pos))


def render(clues, revealed, hit=None):
    image = Image.new("RGB", (SIZE * CELL, SIZE * CELL), BOARD)
    draw = ImageDraw.Draw(image)
    for pos in CELLS:
        left, top, right, bottom = cell_box(pos)
        covered = pos not in revealed
        background = "#963d36" if pos == hit else "#3c536e" if covered else "#233249"
        draw.rectangle((left, top, right, bottom), fill=background, outline=GRID)
        symbol = "*" if pos == hit else "?" if covered else str(clues[pos] or "")
        draw.text((left + CELL / 2, top + CELL / 2), symbol, anchor="mm", font=FONT, fill=INK)
    return image


def main():
    cli = parser(__doc__, "minesweeper.gif", SIZE * SIZE)
    cli.add_argument("--seed", type=int, default=0)
    args = parse_game_args(cli)
    question = (
        "This is Minesweeper. A question mark is a covered cell. Each revealed number "
        "counts mines in the eight neighboring cells, including diagonals. A blank revealed "
        f"cell means zero. The board contains exactly {MINE_COUNT} mines. "
        "Select one covered cell that is safe to reveal."
    )
    opening = {START, *neighbors(START)}
    mine_candidates = [pos for pos in CELLS if pos not in opening]
    mines = set(random.Random(args.seed).sample(mine_candidates, MINE_COUNT))
    clues = {pos: sum(neighbor in mines for neighbor in neighbors(pos)) for pos in CELLS}
    safe_count = len(CELLS) - MINE_COUNT
    revealed = set()
    reveal(START, clues, revealed)

    moves = []
    hit = None
    frames = [replay_frame(render(clues, revealed), "Minesweeper")]
    outcome = "turn limit"
    for turn in range(1, args.max_turns + 1):
        if len(revealed) == safe_count:
            outcome = "solved"
            break
        image = render(clues, revealed)
        covered = [pos for pos in CELLS if pos not in revealed]
        cell, score = choose(args.url, question, image, covered)
        frames.append(replay_frame(image, f"Move {turn}  |  Score {score:.3f}", cell))
        moves.append(dict(turn=turn, row=cell[0] + 1, col=cell[1] + 1, score=score))
        if cell in mines:
            hit, outcome = cell, "hit a mine"
            break
        reveal(cell, clues, revealed)
    if len(revealed) == safe_count:
        outcome = "solved"
    frames.append(replay_frame(render(clues, revealed, hit), f"Minesweeper  |  {outcome}"))
    save_replay(args.output, frames, "minesweeper", outcome, moves, seed=args.seed)


if __name__ == "__main__":
    main()
