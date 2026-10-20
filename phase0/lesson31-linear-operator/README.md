# Lesson 31 — A fused linear operator

This lesson turns matrix multiplication into the operation that appears
throughout a transformer:

```text
linear(input, weight, bias)
```

The implementation performs, in one Metal kernel:

```text
matrix multiplication
        ↓
optional bias addition
        ↓
optional ReLU
```

## What we built

```cpp
enum class LinearActivation {
    None,
    Relu,
};

class LinearOperator {
public:
    explicit LinearOperator(
        const minigpu::Device& device
    );

    RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        const RuntimeTensor& weight,
        const RuntimeTensor* bias = nullptr,
        LinearActivation activation =
            LinearActivation::None,
        std::size_t threads_per_group = 64
    ) const;
};
```

The constructor compiles the Metal kernel once. Every call to `run()` reuses
that compiled pipeline with different tensors, dimensions, and flags.

## 1. What is a linear layer?

Suppose:

```text
input  shape = [M, K]
weight shape = [N, K]
bias   shape = [N]       optional
```

The result has shape:

```text
output shape = [M, N]
```

For one output value:

```text
output[row, column]
    =
    sum over k(
        input[row, k] * weight[column, k]
    )
    + bias[column]
```

Without bias:

```text
output = input @ weightᵀ
```

With bias:

```text
output = input @ weightᵀ + bias
```

The transpose symbol does not mean that this lesson creates a new transposed
weight buffer. The kernel simply reads one row of `weight` for each output
column.

## 2. Why is the weight `[N, K]`?

Hugging Face models are usually loaded through PyTorch-style linear layers.
Their weight is stored as:

```text
[output_features, input_features]
       N                K
```

For example:

```text
input  [tokens, hidden_size]
weight [new_hidden_size, hidden_size]
output [tokens, new_hidden_size]
```

That is why our API accepts `[N,K]`, rather than the mathematical `[K,N]`
layout expected by an ordinary `A @ B` call.

Keeping the real model layout matters. Later, weights loaded from a model do
not need to be transposed just to enter this operator.

## 3. A small example

```text
input [2,3]

  [ 1, 2, 3]
  [-1, 0, 2]

weight [4,3]

  [1, 0, 0]
  [0, 1, 0]
  [0, 0, 1]
  [1, 1, 1]
```

The first output row contains four dot products:

```text
[1,2,3] · [1,0,0] = 1
[1,2,3] · [0,1,0] = 2
[1,2,3] · [0,0,1] = 3
[1,2,3] · [1,1,1] = 6
```

Therefore:

```text
input @ weightᵀ

  [ 1, 2, 3, 6]
  [-1, 0, 2, 1]
```

## 4. Bias is broadcast across rows

For:

```text
bias = [-2, 0.5, -4, -3]
```

the same bias vector is added to every input row:

```text
  [ 1, 2, 3, 6]   + [-2, 0.5, -4, -3]
  [-1, 0, 2, 1]   + [-2, 0.5, -4, -3]

      ↓

  [-1, 2.5, -1,  3]
  [-3, 0.5, -2, -2]
```

The bias has only `N` elements. We do not allocate an `[M,N]` copy of it.
Every output lane reads `bias[column]`.

## 5. Optional ReLU is fused

ReLU is:

```text
relu(x) = max(x, 0)
```

After bias and ReLU:

```text
  [0, 2.5, 0, 3]
  [0, 0.5, 0, 0]
```

A less efficient implementation could launch:

```text
kernel 1: matmul
kernel 2: add bias
kernel 3: ReLU
```

That would create extra dispatches and repeatedly read and write the whole
output tensor.

Our kernel calculates one output scalar in a local variable:

```metal
float sum = dot_product(...);

if (has_bias != 0) {
    sum += bias[column];
}

if (activation != 0) {
    sum = max(sum, 0.0f);
}

output[global_id] = sum;
```

The intermediate result remains lane-local. Only the final value is written
to GPU memory.

This is operator fusion at a small, concrete scale.

## 6. One lane owns one output element

For output `[M,N]`, we launch enough lanes to cover:

```text
M × N values
```

A flat lane ID becomes:

```text
row    = global_id / N
column = global_id % N
```

Then that lane loops over `K`:

```text
lane for output[row,column]

input[row,0] × weight[column,0]
input[row,1] × weight[column,1]
...
input[row,K-1] × weight[column,K-1]
```

Different lanes may read the same input or weight values. That is safe:
they only read shared tensors and write distinct output positions.

No barrier or atomic is needed.

## 7. One reusable kernel handles four modes

Two scalar flags select:

```text
no bias, no activation
bias, no activation
no bias, ReLU
bias and ReLU
```

We do not compile four Metal kernels.

The optional bias argument still needs a valid Metal buffer binding. When
bias is absent, the runtime binds the output buffer as a harmless placeholder
and sets `has_bias = 0`. The kernel therefore never reads through the bias
pointer.

## 8. Layout rule

This first linear operator requires canonical row-major input storage:

```text
input, weight, and bias must report is_contiguous()
```

A contiguous tensor may have a nonzero storage offset. The kernel binds each
offset and adds it to its address calculation.

For a non-contiguous transpose or stepped slice, first use Lesson 28:

```text
view
  ↓ contiguous materialization
compact tensor
  ↓ LinearOperator
output
```

Supporting arbitrary strides inside every matrix kernel is possible, but it
would obscure the main linear-layer idea and make this teaching kernel less
compact.

## 9. Validation performed by the CPU

Before dispatch, `run()` checks:

```text
input rank is 2
weight rank is 2
input K equals weight K
bias rank is 1
bias length equals N
all inputs are contiguous
threads_per_group is nonzero
indexes and launch sizes fit uint32
```

Bad tensor metadata becomes a readable C++ exception instead of an invalid
GPU memory access.

## 10. Output and synchronization

`run()` allocates a fresh compact tensor:

```text
shape   [M,N]
offset  0
strides [N,1]
```

It submits the Metal dispatch and waits for its `Event` before returning.
Therefore this is safe immediately:

```cpp
RuntimeTensor output = linear.run(...);
std::vector<float> values = output.to_host();
```

Later inference-engine layers will keep more work asynchronous and wait only
at real dependency boundaries.

## 11. Is this fast?

It is correct and useful, but intentionally naive.

Each lane independently walks all `K` values. Nearby lanes repeatedly fetch
some of the same input data. A production kernel uses ideas such as:

```text
tiles
threadgroup memory
SIMD-group matrix operations
vectorized loads
half-precision or quantized weights
hardware-specific tuning
```

The important lesson is the operator contract and fusion. We can replace the
kernel implementation later without changing the user-facing API.

## Files

```text
linear.hpp   public API
linear.cpp   validation, Metal source, dispatch
demo.cpp     CPU comparisons, examples, error tests
README.md    lesson guide
```

## Build and run

From `outputs/infer-lab/phase0`:

```bash
mkdir -p work/lesson31

clang++ -std=c++20 \
  -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wsign-conversion \
  lesson14-metal-minigpu/minigpu.mm \
  lesson21-types-shapes-broadcasting/miniir.cpp \
  lesson25-tensor-storage-layout/runtime_tensor.cpp \
  lesson31-linear-operator/linear.cpp \
  lesson31-linear-operator/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson31/demo

./work/lesson31/demo
```

## Main takeaways

1. A linear layer computes `input @ weightᵀ`.
2. Hugging Face-style weights are stored as `[N,K]`.
3. Bias `[N]` is reused for every input row.
4. Bias and activation can be fused into the matmul kernel.
5. One lane owns one output scalar and performs a `K`-length dot product.
6. The same compiled kernel supports optional bias and optional ReLU.
7. This version accepts contiguous f32 tensors and returns `[M,N]`.

## Next lesson

Lesson 32 adds a batch dimension:

```text
[B,M,K] @ [B,K,N] → [B,M,N]
```

That operation will also give us the exact shape pattern needed for
attention's `QKᵀ`.
