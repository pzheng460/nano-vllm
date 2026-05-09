<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, ACL graph, etc.
* ✅ **Multi-Platform Support** - GPU, NPU

## Installation

### GPU Installation

For GPU users, we recommend [uv](https://github.com/astral-sh/uv):

```bash
uv pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

(If you prefer `pip`, `pip install git+https://github.com/GeeeekExplorer/nano-vllm.git` also works.)

#### Editable install with extras (recommended for development)

For developing locally on this fork (e.g. running the `experiments/pangu_lsd/`
benches and trace analysis):

```bash
git clone https://github.com/pzheng460/nano-vllm.git
cd nano-vllm
git checkout latent-sd
```

Then pick one of the two flows below.

##### Option A: uv (fastest)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pip install uv
uv venv                                            # creates .venv with Python in [3.10, 3.13)
uv pip install -e '.[cuda]'                        # core + triton + flash-attn
# uv pip install -e '.[cuda,profile,hf]'           # add ijson + datasets
# uv pip install -e '.[all]'                       # cuda + npu + profile + hf
```

`uv venv` does not install `pip` by default. If you want plain `pip` inside
the venv (e.g. for follow-up installs):

```bash
uv pip install pip
source .venv/bin/activate
pip install -e '.[cuda,hf]'
```

##### Option B: conda

```bash
conda create -n nanovllm python=3.12 -y
conda activate nanovllm
pip install -e '.[cuda]'
# pip install -e '.[cuda,profile,hf]'
```

##### Notes

- `flash-attn` builds from source on first install (~5–15 min on CUDA 12 +
  GCC 11+). Pass `--no-build-isolation` to speed it up:
  ```bash
  uv pip install flash-attn --no-build-isolation
  # or:
  pip install flash-attn --no-build-isolation
  ```
- To pin a specific PyTorch CUDA wheel (e.g. cu128) before pulling the
  rest:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -e '.[cuda]'
  ```
- Sanity-check after install:
  ```bash
  python -c "import nanovllm, flash_attn, safetensors, sentencepiece; print('ok')"
  ```
- `experiments/pangu_lsd/test.sh` auto-detects the interpreter — it prefers
  `$PYTHON` env var, then `./.venv/bin/python` (uv/venv layout), then
  `python` on `PATH` (conda activate). No edits needed.

### Ascend NPU Installation

For Huawei Atlas 800I/800T A2/A3 users, we recommend starting with Docker.

```bash
docker pull quay.io/ascend/vllm-ascend:v0.13.0rc1
```
```bash
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc1
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```
The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in development mode (`pip install -e`) to help developer immediately take place changes without requiring a new installation.

```bash
uv pip install -e .
```
Install the nano-vllm from source (falls back to `pip install -e .` if `uv` is unavailable).

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)