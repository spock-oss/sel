"""Play a maze one move at a time using a running SEL server."""

from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw

from _gameplay import (
    CELL,
    cell_box,
    choose,
    parse_game_args,
    parser,
    replay_frame,
    save_replay,
)

DEFAULT_BOARD = (
    "########",
    "##S#####",
    "##.#####",
    "##.#####",
    "##...G##",
    "########",
    "########",
    "########",
)
QUESTION = (
    "This is a maze. The red disk is the current position and the green disk is the goal. "
    "Dark squares are walls, light squares are open, and yellow dots mark visited squares. "
    "You can move up, down, left, or right through open squares. Visited squares are passable. "
    "Select only the one adjacent square to move to next on the shortest route to the green goal."
)


def neighbors(cell):
    row, col = cell
    return [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]


def read_board(path):
    rows = tuple(path.read_text().splitlines()) if path else DEFAULT_BOARD
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("The board must be a nonempty rectangle.")
    if not (4 <= len(rows) <= 24 and 4 <= len(rows[0]) <= 24):
        raise ValueError("Use a board between 4 and 24 cells in each dimension.")
    tiles = {
        (r, c): char for r, row in enumerate(rows) for c, char in enumerate(row)
    }
    if set("".join(rows)) - set("#.SG"):
        raise ValueError("Use # for walls, . for open cells, S for the start, and G for the goal.")
    starts = [cell for cell, char in tiles.items() if char == "S"]
    goals = [cell for cell, char in tiles.items() if char == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("The board must contain exactly one S and one G.")
    open_cells = {cell for cell, char in tiles.items() if char != "#"}
    frontier = deque(starts)
    reachable = set(starts)
    while frontier:
        for cell in neighbors(frontier.popleft()):
            if cell in open_cells and cell not in reachable:
                reachable.add(cell)
                frontier.append(cell)
    if goals[0] not in reachable:
        raise ValueError("The goal must be reachable from the start.")
    return rows, open_cells, starts[0], goals[0]


def render(rows, current, goal, visited):
    image = Image.new("RGB", (len(rows[0]) * CELL, len(rows) * CELL))
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(rows):
        for col, value in enumerate(line):
            pos = (row, col)
            box = cell_box(pos)
            background = "#293241" if value == "#" else "#f7f6ec"
            draw.rectangle(box, fill=background, outline="#939ca6")
            left, top, right, bottom = box
            if pos == current or pos == goal:
                color = "#24a851" if pos == goal else "#e5383b"
                draw.ellipse((left + 6, top + 6, right - 6, bottom - 6), fill=color)
            elif pos in visited:
                draw.ellipse((left + 11, top + 11, right - 11, bottom - 11), fill="#dfa614")
    return image


def main():
    cli = parser(__doc__, "maze.gif", 64)
    cli.add_argument("--board", type=Path, help="ASCII board using # . S G")
    args = parse_game_args(cli)
    try:
        rows, open_cells, current, goal = read_board(args.board)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))

    moves = []
    visited = {current}
    frames = [replay_frame(render(rows, current, goal, visited), "Maze")]
    outcome = "turn limit"
    for turn in range(1, args.max_turns + 1):
        image = render(rows, current, goal, visited)
        candidates = [pos for pos in neighbors(current) if pos in open_cells]
        next_cell, score = choose(args.url, QUESTION, image, candidates)
        frames.append(replay_frame(image, f"Move {turn}  |  Score {score:.3f}", next_cell))
        moves.append(dict(turn=turn, row=next_cell[0] + 1, col=next_cell[1] + 1, score=score))
        current = next_cell
        visited.add(current)
        if current == goal:
            outcome = "solved"
            break
    frames.append(replay_frame(render(rows, current, goal, visited), f"Maze  |  {outcome}"))
    save_replay(args.output, frames, "maze", outcome, moves, board=rows)


if __name__ == "__main__":
    main()
