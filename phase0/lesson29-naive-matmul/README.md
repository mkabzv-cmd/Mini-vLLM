# Lesson 29 — Naive matrix multiplication

## What we built

```cpp
NaiveMatmul matmul(device);
RuntimeTensor c = matmul.run(queue, a, b);
```

It computes:

```text
A [M,K] × B [K,N] → C [M,N]
```

The class compiles its Metal kernel once. Every call can then reuse that
compiled pipeline with different compact matrices.

`run()` is synchronous:

```text
validate inputs
allocate C
submit the Metal kernel
wait for its Event
return C
```

## What matrix multiplication means

For:

```text
A = [1 2 3]       B = [7  8]
    [4 5 6]           [9 10]
                      [11 12]
```

one output cell is a row of `A` dotted with a column of `B`:

```text
C[0,0] = 1×7 + 2×9 + 3×11 = 58
C[0,1] = 1×8 + 2×10 + 3×12 = 64
```

The full result is:

```text
C = [ 58  64]
    [139 154]
```

Every output cell performs `K` multiply-add steps.

## One lane owns one output cell

The GPU launch is one-dimensional:

```text
global_id:  0    1    2    3
output:    C00  C01  C10  C11
```

The kernel converts a flat lane ID into a row and column:

```metal
row    = global_id / N;
column = global_id % N;
```

Then that lane computes:

```metal
float sum = 0.0f;

for (uint k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + column];
}

C[global_id] = sum;
```

This is called naive because lanes independently fetch every value they
need from the main GPU buffer. Neighboring lanes repeatedly load many of the
same values.

It is still the right first implementation because the ownership is simple:

```text
one lane
   ↓
one output cell
   ↓
no output race
```

## Why the inputs must be compact

This lesson accepts only rank-two `f32` tensors with:

```text
canonical row-major strides
offset = 0
storage size = logical element count
```

The kernel uses direct formulas:

```text
A index = row × K + k
B index = k × N + column
```

Those formulas are not enough for a transpose or stepped view. Lesson 28 can
materialize such a view before matmul:

```text
view → contiguous materialization → matmul
```

Later, the compiler can insert that conversion automatically.

## Shape checking

The inner dimensions must match:

```text
A [M,K]
B [K,N]
```

This is valid:

```text
[5,7] × [7,4] → [5,4]
```

This is invalid:

```text
[5,7] × [6,4]
```

The CPU validates the ranks, layouts, shapes, launch size, and the kernel's
32-bit indexing boundary before submitting work.

## Why this matters for inference

Most transformer computation eventually becomes matrix multiplication:

```text
token activations × weight matrix
attention queries × attention keys
attention probabilities × values
MLP activations × projection weights
```

Real inference systems use heavily optimized matmul kernels. This lesson
creates the small, obviously correct baseline that Lesson 30 will optimize.

## Files worth reading

Read in this order:

1. `naive_matmul.hpp` — the small public API.
2. The Metal string in `naive_matmul.cpp`.
3. `NaiveMatmul::run()` — allocation, bindings, dispatch, and wait.
4. The known `2×3` by `3×2` example in `demo.cpp`.

The remaining demo code is validation and error testing.

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
  outputs/infer-lab/phase0/lesson29-naive-matmul/naive_matmul.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson29-naive-matmul/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson29_naive_matmul

./work/lesson29_naive_matmul
```

## Main takeaways

1. Matmul produces one dot product per output cell.
2. One GPU lane can own one output cell without an output race.
3. The flat lane ID maps to `(row, column)`.
4. Compact row-major layout makes address calculation simple.
5. Shape validation happens before GPU submission.
6. The compiled Metal pipeline is reused across calls.
7. The synchronous API waits before returning CPU-readable output.
8. This simple kernel is the correctness reference for tiled matmul.

## Next lesson

Lesson 30 will let a whole 8×8 threadgroup cooperate:

```text
load reusable A and B tiles into threadgroup memory
barrier
compute an 8×8 output tile
```
