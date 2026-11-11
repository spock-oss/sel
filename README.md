# SEL 27B

[Model weights](https://huggingface.co/spock-oss/sel-qwen-27b) · [Training dataset](https://huggingface.co/datasets/spock-oss/sel-27b-training) · [Research blog](https://spock.one/blog/selection-is-a-model-shape)

SEL scores source positions in text and images against one or more questions. It runs the Qwen backbone once and reads the resulting hidden states through a dense scorer or a recurrent span pointer. The linked pointer can condition a later selection on the earlier one using the same backbone states.

This repository contains the 27B inference and training code. It includes a tiny synthetic dataset for smoke tests. The trained adapter, selection head and full 26-task corpus are separate artifacts.

## Install

Use Linux, Python 3.12 and an NVIDIA CUDA GPU with BF16 support. The tested stack uses the CUDA 13.0 PyTorch build. Use a compatible CUDA toolkit, C++ compiler and Ninja to build FlashAttention 2, or install a wheel built for the same Python, PyTorch and CUDA combination.

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e . packaging setuptools wheel ninja
uv pip install --no-build-isolation flash-attn==2.8.3.post2
```

The loader uses the BF16 snapshot of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0), about 56 GB on disk. It quantizes the frozen language backbone to NF4 at load time and preserves the vision encoder and LM head in BF16. Set `CUDA_VISIBLE_DEVICES` to one GPU. Pass `--base /path/to/snapshot` to use an existing copy. Otherwise the command downloads the pinned snapshot from Hugging Face.

## Run inference

Point `--weights` at the directory containing `adapter_model.safetensors`, `adapter_config.json`, `selection_head.safetensors` and `sel_config.json`.

```sh
CUDA_VISIBLE_DEVICES=0 sel predict --weights .artifacts/sel-27b \
  --text 'This agreement is governed by the laws of Oregon.' \
  --query 'Which state governs the agreement?'
```

Use `--text-file` for a document and `--image` for an image. Both can be supplied together. Repeat `--query` for multiple questions. The command returns per-question scores and selected source indices, with character offsets for text and bounding boxes in source-image coordinates for patches. Use `--mode` to override the learned router, `--threshold` for a custom dense threshold, and `--calibrated` to use thresholds measured on validation. Input is limited to 2,048 tokens including the prompt and visual tokens. Over-length inputs are rejected.

## Prepare and train

```sh
sel prepare --source examples --output .artifacts/demo-data
CUDA_VISIBLE_DEVICES=0 sel train --data .artifacts/demo-data \
  --output .artifacts/demo-run --max-steps 1
CUDA_VISIBLE_DEVICES=0 sel train --data .artifacts/demo-data \
  --output .artifacts/demo-run --resume
```

The source directory provides `train.jsonl`, `val.jsonl`, and `test.jsonl`. Each row names an `artifact_path` and can set `selection_mode` (`dense`, `one_coherent_answer`, or `linked`), `task_id`, `query_first`, and `environment_prompt`. Each record contains the source, questions, verification status, and complete acceptable solutions keyed by question ID. See [text](examples/records/state.json), [image](examples/records/colors.json), and [linked questions](examples/records/train-time.json). Text spans use half-open character offsets. Image annotations use source-pixel coordinates. Keep related problems in the same split when building your manifests.

The training recipe uses rank-16 LoRA, a 256-wide selection head, four examples per optimizer step, and one epoch. The 16 full-attention blocks are bidirectional and the 48 linear-attention blocks remain directional. Optimizer, sampler, RNG, and LR position are saved to `latest.pt` for resume. Validation calibrates thresholds and test reports fixed and calibrated metrics.

## Export and evaluate

```sh
sel export --checkpoint .artifacts/demo-run/latest.pt --output .artifacts/demo-weights
CUDA_VISIBLE_DEVICES=0 sel evaluate --weights .artifacts/demo-weights \
  --data .artifacts/demo-data --split test
```

Export reads a trusted local training checkpoint and writes the inference tensors as safetensors. Download the released bundle and use its directory with `--weights`.

```sh
hf download spock-oss/sel-qwen-27b \
  --include '*.safetensors' adapter_config.json sel_config.json \
  --local-dir .artifacts/sel-27b
```

The [dataset card](https://huggingface.co/datasets/spock-oss/sel-27b-training) shows how to download the public splits and materialize them for `sel prepare`.
