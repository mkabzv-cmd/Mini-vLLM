# Infer Lab

Infer Lab is a correctness-first, miniature AI inference stack built for
learning. It starts with a CPU simulation of GPU concepts, moves to real Apple
Metal execution, then builds compiler, tensor, transformer, Hugging Face model,
generation, and KV-cache layers.

```text
CPU MiniGPU simulator
        ↓
real Metal MiniGPU runtime
        ↓
MiniIR compiler and Metal code generation
        ↓
MiniTensor, storage, layouts, and views
        ↓
matmul and transformer operators
        ↓
small Hugging Face Llama
        ↓
generation, prefill/decode, and KV cache
```

Lessons 10–39 are implemented. Lessons 40 and later are planned in
[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md). A separate path for image diffusion,
step-aware batching, and a shared text/image server is described in
[FUTURE_ROADMAP_DIFFUSION.md](FUTURE_ROADMAP_DIFFUSION.md).

This is an educational implementation, not a production inference engine.

## Platform support

**You do not need an M4 specifically.** The project was developed and verified
on an Apple M4 Pro, but the code does not use M4-only instructions.

The recommended target for the GPU lessons is a recent Apple Silicon Mac with
Metal support. M1, M2, M3, M4, and newer Apple Silicon systems should generally
work, but they have not all been verified in this repository. A Metal-capable
Intel Mac may also work, but is untested.

| Lessons | Execution backend | Requirements |
| --- | --- | --- |
| 10–12 | CPU MiniGPU simulator | C++20 and threads; no GPU |
| 13–19 | Real Apple GPU | macOS, Metal, Foundation, Objective-C++ |
| 20–21 | CPU compiler and IR | C++20; no Metal |
| 22–24 | Compiler through real GPU | macOS and Metal |
| 25–28 | Metal-backed tensor storage and layouts | macOS and Metal |
| 29–34 | Matmul and transformer operators | macOS and Metal |
| 35–36 | Llama forward pass and generation | macOS, Metal, local model files |
| 37 | CPU planner tests or Metal model demo | Depends on the target built |
| 38 | KV-cache storage and append | macOS and Metal; model files not required |
| 39 | Cached Llama decoding | macOS, Metal, local model files |

The real-GPU lessons currently use Apple Metal, not CUDA or ROCm. They do not
run unchanged on Linux or Windows, and an NVIDIA GPU is not required.

Restricted sandboxes, virtual machines, and some CI environments may hide the
Metal device even when the host Mac has one.

## Prerequisites

For the complete Metal path:

- A physical, Metal-capable Mac.
- Xcode Command Line Tools or full Xcode.
- A compiler with C++20 support.
- Python 3.9 or newer for helper scripts; Python 3.10+ is recommended.
- Git.

Install the Apple command-line tools:

```bash
xcode-select --install
```

Verify the main tools:

```bash
xcrun clang++ --version
python3 --version
```

Metal and Foundation are part of the Apple SDK. There is no separate Metal
package to install. Metal Shading Language source is embedded in the programs
and compiled at runtime.

## Clone and prepare

```bash
git clone https://github.com/mkabzv-cmd/AI.git
cd AI
mkdir -p phase0/work
```

`phase0/work` holds generated executables and test output. It is intentionally
ignored by Git.

All commands in this top-level README are run from the repository root.

Some lesson READMEs were originally written while this project lived inside a
larger workspace. If an older command begins with `outputs/infer-lab/`, remove
that prefix when running from this standalone repository.

## Lesson map

| Lessons | Main subject |
| --- | --- |
| [10](phase0/lesson10-minigpu-cpu/README.md)–[12](phase0/lesson12-atomics-races/README.md) | CPU MiniGPU, queues, buffers, events, barriers, reductions, races, and atomics |
| [13](phase0/lesson13-first-metal-kernel/README.md)–[19](phase0/lesson19-reduction-softmax-performance/README.md) | Real Metal kernels, launch geometry, memory, barriers, atomics, reductions, and softmax |
| [20](phase0/lesson20-compiler-foundations/README.md)–[24](phase0/lesson24-minitensor-frontend/README.md) | IR, SSA values, types, shapes, Metal code generation, optimization, fusion, and MiniTensor |
| [25](phase0/lesson25-tensor-storage-layout/README.md)–[28](phase0/lesson28-contiguous-buffer-reuse/README.md) | Tensor storage, strides, views, non-contiguous layouts, materialization, and buffer reuse |
| [29](phase0/lesson29-naive-matmul/README.md)–[32](phase0/lesson32-batched-matmul/README.md) | Naive matmul, tiled matmul, linear layers, and batched matmul |
| [33](phase0/lesson33-transformer-ops/README.md)–[34](phase0/lesson34-attention-transformer-block/README.md) | RMSNorm, RoPE, embeddings, SiLU, attention, and a transformer block |
| [35](phase0/lesson35-huggingface-llama/README.md) | Load and execute a small Hugging Face Llama checkpoint |
| [36](phase0/lesson36-autoregressive-generation/README.md) | Greedy and sampled autoregressive generation |
| [37](phase0/lesson37-prefill-decode/README.md) | Prefill versus token-by-token decode |
| [38](phase0/lesson38-kv-cache-layout/README.md) | Per-layer KV-cache layout, capacity, and GPU append |
| [39](phase0/lesson39-cached-decoding/README.md) | Real cached Llama decoding and generation |

## Quick start: Lesson 10 CPU MiniGPU

This is the simplest starting point and does not use Metal:

```bash
clang++ \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -pthread \
  phase0/lesson10-minigpu-cpu/minigpu.cpp \
  phase0/lesson10-minigpu-cpu/tests.cpp \
  -o phase0/work/lesson10_tests

phase0/work/lesson10_tests
```

Expected final line:

```text
All MiniGPU CPU tests passed
```

## Quick start: Lesson 13 first real Metal kernel

This compiles Objective-C++ host code, compiles an MSL kernel at runtime, and
runs vector addition on the Apple GPU:

```bash
xcrun clang++ \
  -std=c++20 \
  -O0 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fobjc-arc \
  phase0/lesson13-first-metal-kernel/main.mm \
  -framework Foundation \
  -framework Metal \
  -o phase0/work/lesson13_first_metal

phase0/work/lesson13_first_metal
```

Expected final line:

```text
PASS: vector_add executed correctly on the GPU.
```

## Quick start: Lesson 20 compiler foundations

This lesson builds and evaluates the first SSA-like IR. It is CPU-only:

```bash
xcrun clang++ \
  -std=c++20 \
  -O2 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  phase0/lesson20-compiler-foundations/miniir.cpp \
  phase0/lesson20-compiler-foundations/demo.cpp \
  -o phase0/work/lesson20_first_ir

phase0/work/lesson20_first_ir
```

Expected final line:

```text
PASS: builder, SSA-like IR, verifier, printer and evaluator succeeded.
```

## Quick start: Lesson 30 tiled matmul

This runs a real tiled matrix multiplication kernel using threadgroup memory
and barriers:

```bash
xcrun clang++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -Wconversion \
  -Wsign-conversion \
  -fobjc-arc \
  phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  phase0/lesson25-tensor-storage-layout/runtime_tensor.cpp \
  phase0/lesson29-naive-matmul/naive_matmul.cpp \
  phase0/lesson30-tiled-matmul/tiled_matmul.cpp \
  phase0/lesson14-metal-minigpu/minigpu.mm \
  phase0/lesson30-tiled-matmul/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o phase0/work/lesson30_tiled_matmul

phase0/work/lesson30_tiled_matmul
```

## Model setup for Lessons 35–39

Model files are deliberately excluded from Git.

The lessons currently use:

```text
repository: hf-internal-testing/tiny-random-LlamaForCausalLM
revision:   9fb191250dd56d0ba7ec9785a025ed29c03d5998
directory:  models/hf-tiny-random-llama/
```

One way to download that exact snapshot is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install huggingface_hub

hf download \
  hf-internal-testing/tiny-random-LlamaForCausalLM \
  --revision 9fb191250dd56d0ba7ec9785a025ed29c03d5998 \
  --local-dir models/hf-tiny-random-llama
```

For the tokenizer bridge and official Hugging Face reference comparisons:

```bash
python -m pip install transformers torch safetensors
```

The native C++/Metal engine reads the local JSON and safetensors files itself.
PyTorch is used only by the optional reference programs, not by the native
inference implementation.

This checkpoint has random test weights. It verifies loading, GPU execution,
generation, and caching, but it does not generate meaningful language.

See [models/README.md](models/README.md) for the fixture details.

## End-to-end example: Lesson 39 cached generation

After downloading the model, build the final implemented lesson:

```bash
xcrun clang++ \
  -std=c++20 -O2 \
  -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wsign-conversion -fobjc-arc \
  phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  phase0/lesson25-tensor-storage-layout/runtime_tensor.cpp \
  phase0/lesson26-tensor-views/tensor_views.cpp \
  phase0/lesson28-contiguous-buffer-reuse/contiguous.cpp \
  phase0/lesson31-linear-operator/linear.cpp \
  phase0/lesson32-batched-matmul/batched_matmul.cpp \
  phase0/lesson33-transformer-ops/transformer_ops.cpp \
  phase0/lesson34-attention-transformer-block/transformer_block.cpp \
  phase0/lesson35-huggingface-llama/hf_loader.cpp \
  phase0/lesson35-huggingface-llama/hf_llama.cpp \
  phase0/lesson36-autoregressive-generation/generation.cpp \
  phase0/lesson38-kv-cache-layout/kv_cache.cpp \
  phase0/lesson39-cached-decoding/cached_llama.cpp \
  phase0/lesson39-cached-decoding/cached_generation.cpp \
  phase0/lesson14-metal-minigpu/minigpu.mm \
  phase0/lesson39-cached-decoding/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o phase0/work/lesson39_cached_generate
```

Run it with the token IDs for `Hello`:

```bash
phase0/work/lesson39_cached_generate \
  models/hf-tiny-random-llama \
  greedy \
  8 \
  phase0/work/lesson39_ids.txt \
  1 15043
```

Or use the Python tokenizer bridge:

```bash
python3 phase0/lesson36-autoregressive-generation/run_prompt.py \
  models/hf-tiny-random-llama \
  phase0/work/lesson39_cached_generate \
  "Hello" \
  --max-new-tokens 8
```

Python performs tokenization and text decoding. The C++/Metal executable still
performs the actual model inference, KV-cache updates, and token selection.

## How the builds work

There is intentionally no unified CMake or package-manager build yet. Each
lesson shows the source files and libraries it depends on directly.

- `.cpp` files contain ordinary C++ host code.
- `.mm` files contain Objective-C++ used to call Apple Metal APIs.
- MSL kernels are currently stored as source strings and compiled at runtime.
- `-framework Foundation -framework Metal` links the Apple frameworks.
- `-fobjc-arc` enables automatic reference counting for Objective-C objects.

This explicit style makes the dependency hierarchy visible while learning.

## Common problems

### `xcrun` cannot find `clang++`

Install or select Xcode Command Line Tools:

```bash
xcode-select --install
xcrun clang++ --version
```

### No Metal device is available

Run on a physical, Metal-capable Mac outside a restricted sandbox or virtual
machine. CPU-only lessons can still run without Metal.

### A model directory or `model.safetensors` is missing

Download the exact fixture revision described above and confirm:

```bash
ls models/hf-tiny-random-llama
```

### Generated text looks meaningless

That is expected. The included fixture is a tiny model with random weights,
chosen for correctness tests rather than language quality.

### An older lesson command cannot find `outputs/infer-lab`

Remove the `outputs/infer-lab/` prefix. This GitHub repository already starts
at the `infer-lab` directory.

## Current limitations

The implemented model path is intentionally small:

- FP32 execution.
- Batch size one.
- One unsharded safetensors checkpoint.
- Ordinary multi-head attention; no grouped-query attention yet.
- Default RoPE.
- Mostly synchronous execution.
- No quantized weights.
- No continuous batching or paged KV-cache manager yet.
- No CUDA, ROCm, Linux, or Windows GPU backend.

The planned path toward Mini-vLLM, optimized inference, and Mini-Dynamo is in
[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md). The optional diffusion and Mini-Omni
path is in
[FUTURE_ROADMAP_DIFFUSION.md](FUTURE_ROADMAP_DIFFUSION.md).
