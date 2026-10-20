# Lesson 32 — Batched matmul and attention shapes

Lesson 31 multiplied two matrices. Transformer attention needs many
independent matrix multiplications at once.

This lesson adds the batch dimension:

```text
one dispatch
    ↓
matrix product for batch 0
matrix product for batch 1
matrix product for batch 2
...
```

## What we built

```cpp
class BatchedMatmul {
public:
    explicit BatchedMatmul(
        const minigpu::Device& device
    );

    RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& left,
        const RuntimeTensor& right,
        bool transpose_right = false,
        std::size_t threads_per_group = 64
    ) const;
};
```

The Metal kernel is compiled once in the constructor and reused by every
call.

## 1. Ordinary batched matmul

The normal mode accepts:

```text
left   [B, M, K]
right  [B, K, N]
output [B, M, N]
```

For each batch `b`:

```text
output[b] = left[b] @ right[b]
```

There is no mixing between batches:

```text
left[0] × right[0] → output[0]
left[1] × right[1] → output[1]
left[2] × right[2] → output[2]
```

The general formula is:

```text
output[b,m,n]
    =
    sum over k(
        left[b,m,k] * right[b,k,n]
    )
```

## 2. What does `B` mean?

`B` simply means “how many independent matrix products?”

It could represent:

```text
different user requests
different images
different audio samples
different attention heads
batch × attention heads
```

The kernel does not care what the batch number means. It only guarantees that
each batch index uses the matching left and right matrices.

## 3. One lane owns one output scalar

The output contains:

```text
B × M × N
```

values, so a flat GPU lane ID must recover three coordinates:

```text
n = global_id % N

temporary = global_id / N

m = temporary % M
b = temporary / M
```

That lane then calculates:

```text
output[b,m,n]
```

by looping over `K`.

Every lane writes a distinct output index, so no barrier or atomic is needed.

## 4. Why have `transpose_right`?

Sometimes the right tensor is already stored as:

```text
[B, N, K]
```

but mathematically we need:

```text
left @ rightᵀ
```

With:

```cpp
transpose_right = true;
```

the accepted shapes become:

```text
left   [B, M, K]
right  [B, N, K]
output [B, M, N]
```

The dot product changes from:

```text
right[b,k,n]
```

to:

```text
right[b,n,k]
```

No transposed tensor is allocated. Only the right-hand address formula
changes inside the same kernel.

This matters directly for attention.

# Attention, explained through shapes

## 5. What are Q and K?

Self-attention creates three tensors:

```text
Q = queries
K = keys
V = values
```

For now, focus only on queries and keys.

A query asks:

```text
What information is this token looking for?
```

A key describes:

```text
What information does this token offer?
```

Their dot product gives a raw similarity score.

Large positive score:

```text
query and key point in similar directions
```

Smaller or negative score:

```text
they are less similar
```

## 6. The original attention shapes

After splitting hidden features into attention heads, Q and K commonly have:

```text
Q [batch, heads, query_tokens, head_dimension]
K [batch, heads, key_tokens,   head_dimension]
```

Use short names:

```text
B = batch
H = heads
Q = number of query tokens
T = number of key tokens
D = head dimension
```

Then:

```text
queries [B, H, Q, D]
keys    [B, H, T, D]
```

Every batch/head pair must perform its own matrix product.

## 7. Flatten batch and heads

Our teaching operator supports rank three, so combine the first two axes:

```text
batch_heads = B × H
```

The same logical data can be viewed as:

```text
queries [B×H, Q, D]
keys    [B×H, T, D]
```

Now these names match batched matmul:

```text
batched-matmul B = B×H
batched-matmul M = Q
batched-matmul N = T
batched-matmul K = D
```

The letter reuse can be confusing. Written with descriptive names:

```text
[batch_heads, query_tokens, head_dimension]
                    @
transpose(
  [batch_heads, key_tokens, head_dimension]
)
                    ↓
[batch_heads, query_tokens, key_tokens]
```

That output contains one score for every:

```text
attention head
query token
key token
```

## 8. Why is K transposed?

Take one attention head:

```text
queries [query_tokens, D]
keys    [key_tokens, D]
```

Each row is one `D`-element vector.

We want every query row dotted with every key row:

```text
score[q,t]
    =
    sum over d(
        query[q,d] * key[t,d]
    )
```

Matrix multiplication expects its right-side reduction dimension first.
Mathematically:

```text
[Q,D] @ [D,T] → [Q,T]
```

but keys are naturally stored `[T,D]`. Therefore:

```text
Q @ Kᵀ
```

Our `transpose_right=true` mode reads `key[t,d]` directly. It gets the effect
of `Kᵀ` without creating a physical `[D,T]` copy.

## 9. Concrete attention example

Suppose one head contains:

```text
queries [3,2]

  [1,0]
  [0,1]
  [1,1]

keys [2,2]

  [1,0]
  [1,1]
```

There are three query tokens and two key tokens, so the score matrix is
`[3,2]`:

```text
query [1,0] against both keys → [1,1]
query [0,1] against both keys → [0,1]
query [1,1] against both keys → [1,2]

scores

  [1,1]
  [0,1]
  [1,2]
```

The demo runs two such head-batches in one Metal dispatch.

## 10. These are only raw attention scores

Full scaled dot-product attention is:

```text
scores = Q @ Kᵀ
scores = scores / sqrt(head_dimension)
scores = scores + mask
probabilities = softmax(scores)
output = probabilities @ V
```

Lesson 32 implements only:

```text
Q @ Kᵀ
```

It does not yet:

```text
scale
apply a causal mask
run softmax
multiply by V
```

Separating the stages lets us understand and test each one before fusing
them.

## 11. Connection to KV-cache decoding

During token-by-token generation:

```text
query_tokens = 1
key_tokens   = all cached tokens so far
```

The important shapes become:

```text
new query    [batch_heads, 1, D]
cached keys  [batch_heads, cached_tokens, D]

Q @ Kᵀ
    ↓
[batch_heads, 1, cached_tokens]
```

That single row says how strongly the new token attends to every cached
token.

Later KV-cache lessons will reuse this exact meaning, although their memory
layouts and kernels will become more specialized.

## 12. Layout and validation rules

This compact lesson accepts contiguous row-major f32 tensors. A contiguous
tensor may have a nonzero offset, which the kernel handles.

Before dispatch, the CPU checks:

```text
both tensors have rank 3
batch sizes match
reduction dimensions K match
storage layouts are contiguous
threads_per_group is nonzero
indexes and launch sizes fit uint32
```

In normal mode:

```text
right K = right.shape[1]
right N = right.shape[2]
```

In transpose-right mode:

```text
right N = right.shape[1]
right K = right.shape[2]
```

Non-contiguous views should first be materialized by Lesson 28.

## 13. Output and synchronization

`run()` returns fresh compact storage:

```text
shape   [B,M,N]
offset  0
strides [M×N,N,1]
```

It waits for the Metal `Event` before returning, so the CPU can immediately
read the result.

The input tensors are read-only and remain unchanged.

## 14. Performance boundary

This is a correct naive batched kernel:

```text
one lane
    ↓
one output value
    ↓
loop through K
```

It does not yet use tiled threadgroup memory or Apple GPU matrix
instructions. Production attention kernels also fuse several later stages
and carefully arrange Q, K, V, and KV-cache memory.

The simple implementation gives us the right semantics first:

```text
which matrices interact
which dimensions reduce
what every score means
```

## Files

```text
batched_matmul.hpp   public API
batched_matmul.cpp   validation, Metal kernel, dispatch
demo.cpp             generic batches and QKᵀ attention
README.md            lesson guide
```

## Build and run

From `outputs/infer-lab/phase0`:

```bash
mkdir -p work/lesson32

clang++ -std=c++20 \
  -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wsign-conversion \
  lesson14-metal-minigpu/minigpu.mm \
  lesson21-types-shapes-broadcasting/miniir.cpp \
  lesson25-tensor-storage-layout/runtime_tensor.cpp \
  lesson32-batched-matmul/batched_matmul.cpp \
  lesson32-batched-matmul/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson32/demo

./work/lesson32/demo
```

## Main takeaways

1. Batched matmul performs one independent matrix product per batch index.
2. Normal mode computes `[B,M,K] @ [B,K,N]`.
3. Transpose-right mode consumes `[B,N,K]` without allocating a transpose.
4. Attention flattens `batch × heads` into the batched-matmul dimension.
5. `Q @ Kᵀ` produces one raw score for every query/key token pair.
6. Decode attention commonly has `M = 1` and `N = cached_tokens`.
7. Scaling, masking, softmax, and multiplication by V are later stages.

## Next step

The next attention layer can build on these raw scores:

```text
scale → causal mask → softmax → multiply by V
```
