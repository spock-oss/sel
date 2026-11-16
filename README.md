# SEL

SEL selects relevant text tokens and image patches in a single model forward
pass. Send a question and a source to the server. It returns scores with
character ranges for text and pixel boxes for images.

[SEL Qwen 0.8B](https://huggingface.co/spock-oss/sel-qwen-0.8b) ·
[SEL Qwen 9B](https://huggingface.co/spock-oss/sel-qwen-9b) ·
[Selection as a model shape](https://spock.one/blog/selection-is-a-model-shape)

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

## API

`POST /select` accepts JSON with a `question` and one or both of `text`
and `image_base64`. The optional `threshold` sets the selection cutoff.
Each result unit contains a score, a selection flag, and its position in
the source. Inputs are limited to 2,048 tokens, including the question,
prompt formatting, and image patches.

## Training

The [training dataset](https://huggingface.co/datasets/spock-oss/sel-data)
contains the packed inputs for both models. The
[reproduction guide](https://github.com/spock-oss/spock-research/blob/main/experiments/sel-9b-reproduction.md)
pins the corpus, base models, and training environment. Reproducing training
requires access to the dataset and Spock Research.
