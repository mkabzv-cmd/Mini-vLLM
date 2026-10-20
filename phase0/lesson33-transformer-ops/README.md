# Lesson 33 — Transformer operations

This lesson adds four real Metal operations that appear around a transformer’s
matrix multiplications:

```text
token IDs ── embedding ── hidden vectors

hidden vectors ── RMSNorm ── normalized vectors

Q and K vectors ── RoPE ── position-aware Q and K

feed-forward gate ── SiLU ── activated gate
```

The implementations are deliberately small, but they execute on the Apple
GPU and return real `RuntimeTensor` objects.

## The API

```cpp
TransformerOps ops(device);

RuntimeTensor hidden =
    ops.embedding(queue, token_ids, embedding_table);

RuntimeTensor normalized =
    ops.rms_norm(queue, hidden, norm_weight);

RuntimeTensor rotated =
    ops.rope(queue, q_or_k, start_position);

RuntimeTensor activated =
    ops.silu(queue, gate);
```

Every method is synchronous:

```text
validate
allocate output
submit Metal kernel
wait for Event
return CPU-readable RuntimeTensor
```

Later, an inference engine should keep commands asynchronous and wait only at
real dependency boundaries. Synchronous methods make the first implementation
easier to inspect and test.

## Where these operations appear

A simplified Llama-like block looks like:

```text
token IDs
   │
   ▼
embedding lookup
   │
   ▼
hidden state
   │
   ├── RMSNorm ── Q/K/V projections
   │                    │
   │                    ├── RoPE(Q)
   │                    └── RoPE(K)
   │                           │
   │                       attention
   │
   └── residual connection
           │
           ▼
        RMSNorm
           │
           ▼
   feed-forward projections
           │
           ├── SiLU(gate)
           │
           └── multiply with up projection
```

This lesson does not yet assemble the whole block. It creates reusable pieces
that a later layer can compose.

## Input rule: compact f32 tensors

The tensor inputs must have:

```text
f32 values
canonical row-major strides
offset = 0
```

In this lesson, “compact” describes the logical layout. The backing allocation
may have extra capacity, but the tensor’s first logical value must begin at
storage index zero.

A transpose or stepped slice must first be materialized by Lesson 28.

---

## 1. Embedding lookup

### What it does

The embedding table contains one learned vector for every vocabulary entry:

```text
table shape = [vocabulary_size, hidden_size]

row 0  → vector for token 0
row 1  → vector for token 1
row 2  → vector for token 2
...
```

For token IDs:

```text
[4, 1, 4]
```

the output is:

```text
[table row 4,
 table row 1,
 table row 4]
```

Its shape is:

```text
[token_count, hidden_size]
```

### GPU indexing

One GPU lane copies one output number:

```metal
token_index  = global_id / hidden_size;
hidden_index = global_id % hidden_size;
token_id     = token_ids[token_index];

output[global_id] =
    table[token_id * hidden_size + hidden_index];
```

Different lanes can copy different columns of the same selected row.

### Why token IDs temporarily use f32

Token IDs are integers, but the current MiniGPU `Buffer` stores `float`.

The public API still accepts the correct host type:

```cpp
std::vector<std::uint32_t> token_ids;
```

Internally the lesson creates:

```text
temporary contiguous f32 tensor
```

Before doing so, every ID is checked:

```text
ID is inside the vocabulary
ID can be represented exactly as f32
```

For example, `16,777,217` cannot be represented exactly by binary32 and is
rejected rather than silently changing to a neighboring integer.

The Metal kernel converts the exact temporary value back to `uint`.

This is a clearly isolated teaching workaround. A production runtime would
add native integer buffers.

---

## 2. RMSNorm

### What it does

For one hidden vector `x` with `H` values:

```text
mean_square = (x₀² + x₁² + ... + xₕ₋₁²) / H

inverse_rms = 1 / sqrt(mean_square + epsilon)

outputᵢ = xᵢ × inverse_rms × weightᵢ
```

RMSNorm controls the scale of a hidden vector. Unlike LayerNorm, it does not
subtract the mean.

The input may have any positive rank. The final dimension is treated as the
hidden dimension:

```text
input  [tokens, hidden]
weight [hidden]
```

or:

```text
input  [batch, tokens, hidden]
weight [hidden]
```

Every combination before the final dimension is one row.

### One threadgroup per row

The kernel always launches:

```text
one 64-thread group for each row
```

Each lane sums a strided subset:

```text
lane 0: columns 0, 64, 128, ...
lane 1: columns 1, 65, 129, ...
...
```

Then all 64 partial sums are reduced in threadgroup memory:

```text
64 partial sums
       ↓
32 additions
       ↓ barrier
16 additions
       ↓ barrier
 8
       ↓
 4
       ↓
 2
       ↓
 1 row sum
```

The barriers matter because lane 0 must not read a partial sum before its
producer lane has written it.

After the reduction, every lane uses the same `inverse_rms` and writes its
strided output columns.

### Why `bind_f32` is used

`epsilon` is a scalar floating-point kernel argument:

```cpp
dispatch.bind_f32(4, epsilon);
```

MiniGPU copies those four bytes into the Metal command, just as `bind_u32`
copies integer metadata.

---

## 3. Llama-style split-half RoPE

### Why position information is needed

Attention receives token vectors. Without a position operation, it cannot
distinguish:

```text
dog bites man
```

from:

```text
man bites dog
```

RoPE rotates pairs of Q and K components by an angle determined by the token
position. Attention dot products then contain relative-position information.

### Input shape

```text
[tokens, heads, head_dimension]
```

`head_dimension` must be even.

### “Split-half” pairing

For:

```text
[x0, x1, x2, x3]
```

Llama-style split-half pairing is:

```text
x0 pairs with x2
x1 pairs with x3
```

It is not:

```text
x0 with x1
x2 with x3
```

For pair index `i`:

```text
frequencyᵢ = 1 / theta^(2i / head_dimension)
angle      = position × frequencyᵢ

first'  = first × cos(angle) - second × sin(angle)
second' = second × cos(angle) + first  × sin(angle)
```

One lane owns one pair and writes both rotated values, so no two lanes write
the same output location.

### `start_position`

During prompt processing, positions may begin at zero.

During token-by-token decoding, the new token may begin later:

```cpp
RuntimeTensor rotated_key =
    ops.rope(queue, key, cached_token_count);
```

The kernel uses:

```text
position = start_position + token_index
```

The host validates that the final position fits the kernel’s 32-bit indexing
model.

---

## 4. SiLU

SiLU is the smooth activation used in Llama’s feed-forward gate:

```text
SiLU(x) = x × sigmoid(x)
        = x / (1 + exp(-x))
```

Examples:

```text
large negative x → value near zero
x = 0            → 0
large positive x → value near x
```

It is elementwise, so one lane performs:

```metal
output[global_id] =
    value / (1.0f + exp(-value));
```

In a SwiGLU feed-forward block it is used approximately as:

```text
SiLU(gate_projection(x)) × up_projection(x)
```

That final multiply and the surrounding matrix multiplications are separate
operations.

---

## Shape summary

| Operation | Input | Other input | Output |
|---|---|---|---|
| Embedding | token IDs `[T]` | table `[V,H]` | `[T,H]` |
| RMSNorm | `[...,H]` | weight `[H]` | `[...,H]` |
| RoPE | `[T,heads,D]` | position and theta | `[T,heads,D]` |
| SiLU | any compact shape | none | same shape |

## Safety checks

The host checks:

```text
all RuntimeTensor inputs are compact
thread counts are positive
launched one-dimensional indexes fit uint32
token list is not empty
embedding table has rank two
token IDs are exact in f32 and inside the vocabulary
RMSNorm input has a final dimension
RMSNorm weight matches that final dimension
RMSNorm epsilon is finite and positive
RoPE input has rank three
RoPE head dimension is even
RoPE theta is finite and positive
RoPE positions fit uint32
```

Padded elementwise lanes return before accessing memory.

RMSNorm does not use padded groups: it intentionally launches exactly one
fixed 64-thread group per row.

## What the demo proves

The demo compares every GPU output with an independent CPU reference:

```text
repeated-token embedding lookup
RMSNorm with hidden size 4
RMSNorm with hidden size 70, requiring lane-strided work
split-half RoPE with nonzero start position
SiLU over negative and positive values
```

It also checks the main validation errors, including an out-of-vocabulary ID,
an ID that is not exactly representable in f32, an incorrect RMSNorm weight,
an odd RoPE head dimension, and non-compact inputs.

## Files

```text
transformer_ops.hpp  public API
transformer_ops.cpp  four Metal kernels and host validation
demo.cpp             CPU references and tests
README.md            this explanation
```

## Build and run

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
  outputs/infer-lab/phase0/lesson33-transformer-ops/transformer_ops.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson33-transformer-ops/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson33_transformer_ops

./work/lesson33_transformer_ops
```

## Main takeaways

1. Embedding lookup turns integer token identities into learned vectors.
2. RMSNorm needs a reduction across each hidden row.
3. RoPE rotates split-half Q/K pairs using token-dependent angles.
4. SiLU is an elementwise gate activation.
5. These different operations can share one `RuntimeTensor` and MiniGPU
   execution model.
6. The current f32 token-ID tensor is a temporary limitation, not a preferred
   production representation.
7. All lesson methods wait before returning, so their outputs are immediately
   safe for CPU inspection.

## Deliberately deferred

```text
native uint32 buffers
f16 and mixed precision
batched RoPE
precomputed sine/cosine tables
fused RMSNorm
fused SwiGLU
asynchronous TransformerOps methods
kernel specialization and autotuning
```

