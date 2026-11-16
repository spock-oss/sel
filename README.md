# SEL

SEL selects positions in text and images in a single model forward pass.
Give it a question and a source. It scores the original text tokens or image
patches and returns their positions. Use it to highlight evidence in a
document or select a region in an image.

[SEL Qwen 9B](https://huggingface.co/spock-oss/sel-qwen-9b) · [Selection as a model shape](https://spock.one/blog/selection-is-a-model-shape)

## Models

| Model | How to run it |
| --- | --- |
| [SEL Qwen 9B](https://huggingface.co/spock-oss/sel-qwen-9b) | Use `sel predict` below for single-question dense selection. |
| [SEL Qwen 0.8B](https://huggingface.co/spock-oss/sel-qwen-0.8b) | Use the runner and instructions in the model repository. |

## Install

The 9B CLI uses Linux, Python 3.12, and a CUDA GPU with BF16 support. The
commands below use PyTorch built for CUDA 13.2. Install the matching CUDA
toolkit and a C++ compiler to build FlashAttention 2, or use a wheel built
for your Python, PyTorch, and CUDA versions.

```sh
git clone https://github.com/spock-oss/sel.git
cd sel
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.15.0 torchvision==0.29.2 \
  --index-url https://download.pytorch.org/whl/cu132
uv pip install -e . packaging setuptools wheel ninja
uv pip install --no-build-isolation flash-attn==2.8.3.post2
```

## Run SEL 9B

Download the LoRA adapter, dense selection head, and their configuration files.

```sh
hf download spock-oss/sel-qwen-9b \
  --include '*.safetensors' adapter_config.json sel_config.json \
  --local-dir .artifacts/sel-qwen-9b

CUDA_VISIBLE_DEVICES=0 sel predict --weights .artifacts/sel-qwen-9b \
  --text 'This agreement is governed by the laws of Oregon.' \
  --query 'Which state governs the agreement?'
```

Use `--text-file` for a document or `--image` for an image. You can supply
both for a multimodal source. The CLI puts the question before the source
and supports up to **2,048 tokens** per input, including the question, prompt
formatting, and any image patches.

The JSON result contains scores and selected source indices. Each source
unit includes character offsets for text or a bounding box in the original
image coordinates. Default selection thresholds are **0.4** for text and
**0.3** for images or mixed inputs. Use `--threshold` to adjust the cutoff.

The CLI downloads the pinned [Qwen3.5-9B base](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
and quantizes the frozen language weights to NF4 at load time. The vision
encoder runs in BF16. Set `CUDA_VISIBLE_DEVICES` to one GPU and use
`--base /path/to/snapshot` to load an existing copy of the base.

## Reproduce training

The 9B model was trained from the pretrained base with rank-32 LoRA and a
dense selection head. The dataset contains 57,927 training examples, 2,406
validation examples, and 3,136 test examples. Each example has one question
and one reference selection mask.

The [training dataset](https://huggingface.co/datasets/spock-oss/sel-data)
contains the exact packed inputs, manifests, and image assets. The
[reproduction guide](https://github.com/spock-oss/spock-research/blob/main/experiments/sel-9b-reproduction.md)
pins the data revision, base model, and training environment. Reproducing
the training run requires access to the dataset and Spock Research.
