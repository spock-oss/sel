# SEL

SEL selects relevant text tokens and image patches in a single model forward
pass. Send a question and a source to the server. It returns scores with
character ranges for text and pixel boxes for images.

[SEL Qwen 0.8B](https://huggingface.co/spock-oss/sel-qwen-0.8b) ·
[SEL Qwen 9B](https://huggingface.co/spock-oss/sel-qwen-9b) ·
[Selection as a model shape](https://spock.one/blog/selection-is-a-model-shape)

## Examples

### Maze

Select every square on the shortest path from red to green.

<img src="examples/maze.png" alt="SEL highlights the path connecting the red start and green goal" width="50%">

### SWE-Gym

Select the existing line of code that needs to change to fix the reported issue.

<img src="examples/swe-gym.png" alt="SEL highlights the line that appends the record list instead of adding its entries individually" width="50%">

### HotpotQA

Which former tennis player played for the United States, Iván Molina or Alex O’Brien? Select the supporting sentences.

<img src="examples/hotpotqa.png" alt="SEL highlights the sentences about O’Brien and Molina" width="50%">

### Game

What can I use to climb?

<img src="examples/game.png" alt="SEL highlights the ladder overhead in LIMBO" width="50%">

Sources include [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) and [Moto](https://github.com/getmoto/moto), [HotpotQA](https://hotpotqa.github.io/) ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), and [LIMBO](https://playdead.com/games/limbo/) by Playdead.

[Interactive examples](https://spock.one/blog/selection-is-a-model-shape#sel-examples)

## Quickstart

On Linux with an NVIDIA GPU, install the tested vLLM runtime.

```sh
git clone https://github.com/spock-oss/sel.git
cd sel
uv venv --python 3.14
source .venv/bin/activate
uv pip install --torch-backend=auto 'vllm==0.33.0' 'transformers==5.20.2' \
  fastapi uvicorn pillow huggingface-hub safetensors

python serve.py spock-oss/sel-qwen-0.8b
```

`serve.py` downloads the adapter, dense selection head, and pinned Qwen
base. Use `python serve.py spock-oss/sel-qwen-9b --fp8` for 9B on an
FP8-capable GPU. The server defaults to `127.0.0.1:8000`.

In another terminal, activate the same environment and run:

```sh
python example.py
```

The client sends a text request, an image request, and a mixed request.
It generates the example image locally.

## Gameplay replays

With the SEL server running, play one move at a time and save an animated GIF:

```sh
python play_maze.py --output replays/maze.gif
python play_minesweeper.py --seed 0 --output replays/minesweeper.gif
```

Each turn sends the current board as an image to `/select`. The script
plays the square with the highest patch score among legal moves, regardless
of the server's selection cutoff. The GIF highlights each chosen cell
before the board updates. The accompanying JSON logs scores, one-based
row and column coordinates, and the outcome.

The maze allows revisiting squares and marks them with yellow dots.
Use `--board my-maze.txt` for a rectangular ASCII board with `#` walls,
`.` open squares, `S` start, and `G` goal. Minesweeper starts with a
safe opening and plays among the covered squares. A mine ends the game.
Both scripts accept `--url` and `--max-turns`.

## API

`POST /select` accepts JSON with a `question` and one or both of `text`
and `image_base64`. The optional `threshold` sets the selection cutoff.
Each result unit contains a score, a selection flag, and its position in
the source. Inputs are limited to 2,048 tokens, including the question,
prompt formatting, and image patches.
