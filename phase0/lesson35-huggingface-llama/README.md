# Lesson 35 — Load and run a Hugging Face Llama

## Result

Our C++/Metal runtime now loads and executes a real Hugging Face checkpoint:

```text
hf-internal-testing/tiny-random-LlamaForCausalLM
revision 9fb191250dd56d0ba7ec9785a025ed29c03d5998
```

The checkpoint is deliberately tiny:

```text
parameters             1,032,272
weight file            4,131,280 bytes
dtype                  f32
transformer layers     2
hidden size            16
attention heads        4
head dimension         4
intermediate size      64
vocabulary size        32,000
```

It uses random test weights. It will not produce meaningful language.

The purpose is more important:

```text
real Hugging Face config
real safetensors file
real Llama tensor names
real model architecture
real Metal execution
```

## End-to-end path

For the prompt:

```text
Hello
```

Hugging Face's tokenizer produces:

```text
[1, 15043]
```

Our runtime then performs:

```text
token IDs
   ↓
embedding lookup
   ↓
transformer block 0
   ↓
transformer block 1
   ↓
final RMSNorm
   ↓
LM-head linear projection
   ↓
logits [2, 32000]
```

The final row contains 32,000 scores: one score for every possible next
token.

## What came from Hugging Face?

The model directory is:

```text
outputs/infer-lab/models/hf-tiny-random-llama/
```

Important files:

```text
config.json          architecture and dimensions
model.safetensors    learned parameter tensors
tokenizer.json       prompt-to-token rules
tokenizer.model      tokenizer vocabulary/model
```

Hugging Face supplies the architecture description and trained—or in this
case randomly initialized—numbers.

Our code supplies the execution machinery.

## What we implemented

### 1. A miniature JSON parser

`config.json` is ordinary JSON.

The loader reads fields such as:

```text
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
rms_norm_eps
rope_theta
vocab_size
```

It rejects variants that our current runtime cannot execute correctly.

### 2. A direct safetensors reader

A safetensors file is approximately:

```text
8-byte header length
        ↓
JSON tensor index
        ↓
raw tensor bytes
```

One header entry resembles:

```json
{
  "dtype": "F32",
  "shape": [32000, 16],
  "data_offsets": [0, 2048000]
}
```

The loader validates:

```text
header and file boundaries
duplicate JSON keys
tensor names
f32 dtype
shape multiplication
expected byte count
non-overlapping data ranges
missing gaps or trailing bytes
```

It then converts the raw bytes into host `float` values and uploads them into
`RuntimeTensor` storage.

### 3. Hugging Face weight mapping

The checkpoint names map directly onto our Lesson 34 block:

| Hugging Face name | Our field |
|---|---|
| `input_layernorm.weight` | attention RMSNorm |
| `self_attn.q_proj.weight` | query weight |
| `self_attn.k_proj.weight` | key weight |
| `self_attn.v_proj.weight` | value weight |
| `self_attn.o_proj.weight` | attention output weight |
| `post_attention_layernorm.weight` | feed-forward RMSNorm |
| `mlp.gate_proj.weight` | gate weight |
| `mlp.up_proj.weight` | up weight |
| `mlp.down_proj.weight` | down weight |

The model-level tensors are:

```text
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

### 4. The model runner

`HuggingFaceLlama::load()`:

```text
parse config
open safetensors
load embeddings
load every layer's weights
load final norm
load LM head
validate all tensor shapes
```

`HuggingFaceLlama::forward()`:

```cpp
hidden = embedding(token_ids);

for (each transformer layer) {
    hidden = transformer_block(hidden, layer_weights);
}

hidden = final_rms_norm(hidden);
logits = linear(hidden, lm_head_weight);
```

This is a complete forward pass for the supported Llama architecture.

## How correctness was checked

`reference.py` loads the same config and weights through the official
Hugging Face Transformers implementation.

Both implementations receive:

```text
[1, 15043]
```

Hugging Face writes its last-token logits into a small binary reference file.
The C++ demo compares all 32,000 values.

Observed result on the Apple M4 Pro:

```text
top-5 IDs, Hugging Face:
[1789, 3038, 30564, 14836, 28241]

top-5 IDs, our Metal runtime:
[1789, 3038, 30564, 14836, 28241]

maximum absolute logit difference:
4.47034836e-08

mean absolute logit difference:
6.27066575e-09
```

This is strong evidence that the embedding, RMSNorm, RoPE, attention,
SwiGLU, residual connections, final normalization, and LM head are connected
correctly.

## Build and run

Run these commands from the workspace root.

### Generate official reference logits

```bash
python3 \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/reference.py \
  outputs/infer-lab/models/hf-tiny-random-llama \
  outputs/infer-lab/phase0/work/lesson35_reference_last_logits.f32
```

### Build the model runner

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
  outputs/infer-lab/phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  outputs/infer-lab/phase0/lesson25-tensor-storage-layout/runtime_tensor.cpp \
  outputs/infer-lab/phase0/lesson26-tensor-views/tensor_views.cpp \
  outputs/infer-lab/phase0/lesson28-contiguous-buffer-reuse/contiguous.cpp \
  outputs/infer-lab/phase0/lesson31-linear-operator/linear.cpp \
  outputs/infer-lab/phase0/lesson32-batched-matmul/batched_matmul.cpp \
  outputs/infer-lab/phase0/lesson33-transformer-ops/transformer_ops.cpp \
  outputs/infer-lab/phase0/lesson34-attention-transformer-block/transformer_block.cpp \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/hf_loader.cpp \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/hf_llama.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o outputs/infer-lab/phase0/work/lesson35_hf_llama
```

### Execute our model and compare

```bash
outputs/infer-lab/phase0/work/lesson35_hf_llama \
  outputs/infer-lab/models/hf-tiny-random-llama \
  outputs/infer-lab/phase0/work/lesson35_reference_last_logits.f32
```

## Files worth reading

Read them in this order:

1. `hf_llama.hpp` — the small public model API.
2. `HuggingFaceLlama::forward()` in `hf_llama.cpp`.
3. `HuggingFaceLlama::load()` and `load_layer()`.
4. `hf_loader.hpp` — the loader's public API.
5. The high-level safetensors parsing code in `hf_loader.cpp`.
6. `demo.cpp` — full-logit comparison.

The internal JSON parser is intentionally thorough but not required reading
right now.

## Current limitations

This first loader/runtime intentionally supports:

```text
one unsharded safetensors file
f32 weights
ordinary multi-head attention
default RoPE
batch size one
no padding mask
full-sequence forward pass
```

It does not yet support:

```text
f16 or bfloat16 weights
sharded checkpoint index files
grouped-query attention
RoPE scaling variants
KV cache
token-by-token cached decoding
continuous batching
paged KV cache
```

The bundled checkpoint is random and therefore cannot generate meaningful
language. A trained checkpoint can run only if it fits the f32, unsharded,
ordinary-multi-head architecture boundary listed above.

## Main takeaway

The Hugging Face model did not bring its own execution engine.

It brought:

```text
configuration
weights
tokenizer data
```

Our miniature stack performed the actual inference.
