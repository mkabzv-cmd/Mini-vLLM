# Lesson 30 — Tiled matrix multiplication

## What changed

Lesson 29 assigned one output cell to one independent lane.

Lesson 30 still gives each lane one output cell, but 64 lanes now cooperate:

```text
one threadgroup
      ↓
one 8×8 output tile
      ↓
64 lanes
```

They cache small pieces of `A` and `B` in fast threadgroup memory before
using them.

```cpp
TiledMatmul matmul(device);
RuntimeTensor c = matmul.run(queue, a, b);
```

The accepted tensors are the same as Lesson 29:

```text
compact f32 A [M,K]
compact f32 B [K,N]
```

## Why tiles help

Consider eight neighboring output cells in one row:

```text
C[row,0] C[row,1] ... C[row,7]
```

All eight calculations need the same values from row `A[row,:]`.

The naive kernel loads those values repeatedly from the main buffer. The
tiled kernel loads an 8×8 block once into threadgroup memory, then lets many
lanes reuse it.

For each part of the `K` dimension:

```text
A tile [8,8] ──┐
                ├── 64 lanes compute part of C tile [8,8]
B tile [8,8] ──┘
```

The full dot product may require several tile rounds:

```text
K = 19

round 0: k = 0..7
round 1: k = 8..15
round 2: k = 16..18 plus zero padding
```

Each lane keeps its partial `sum` in a private variable across the rounds.

## How one threadgroup maps to output

The queue launches:

```text
ceil(M / 8) × ceil(N / 8) threadgroups
```

The MiniGPU launch is one-dimensional, so the kernel converts its flat group
ID into a tile row and tile column.

Inside a group:

```text
local_id 0..63

local_row    = local_id / 8
local_column = local_id % 8
```

For example:

```text
lane 0  → local output [0,0]
lane 1  → local output [0,1]
lane 8  → local output [1,0]
lane 63 → local output [7,7]
```

## The two barriers

Every `K` tile round has two phases.

### 1. Produce, then consume

Each lane writes one `A` value and one `B` value into threadgroup memory.

```metal
a_tile[local_id] = ...;
b_tile[local_id] = ...;

threadgroup_barrier(mem_flags::mem_threadgroup);
```

The first barrier means:

```text
do not read the shared tiles
until all 64 lanes have written their slots
```

### 2. Consume, then overwrite

After every lane has used the current tiles:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

The second barrier means:

```text
do not overwrite the shared tiles for the next round
until all 64 lanes have finished reading this round
```

Both barriers are unconditional. Every lane reaches both barriers in the
same order, including lanes outside the real matrix edges.

## Edge padding

Real dimensions are rarely exact multiples of eight.

For:

```text
A [9,10] × B [10,11]
```

the launch covers:

```text
M capacity = 16
N capacity = 16
K rounds   = 2, covering capacity 16
```

Some lanes therefore represent nonexistent rows, columns, or `K` values.
They must not return early because later barriers need all 64 lanes.

Instead, an invalid load writes zero into the shared tile:

```metal
a_tile[local_id] =
    valid_a_position ? a[...] : 0.0f;
```

Zero contributes nothing to the sum. After all tile rounds, only lanes with
a valid output coordinate store a result:

```metal
if (row < M && column < N) {
    output[row * N + column] = sum;
}
```

The conditional store occurs after every barrier, so it cannot cause barrier
divergence.

## Correctness checks

The demo compares three implementations:

```text
CPU reference
Lesson 29 NaiveMatmul
Lesson 30 TiledMatmul
```

It tests:

```text
small matrices
exact 8×8 tiles
multiple tiles
M, N, and K edge padding
a one-cell output
invalid ranks, shapes, and layouts
```

## Timing

The demo warms up both kernels, records seven synchronous calls, and prints
the median.

The measured call includes:

```text
output allocation
command encoding and submission
GPU execution
Event wait
```

There is deliberately no assertion that tiled must be faster. Performance
depends on matrix size, launch overhead, compiler decisions, and hardware.
Timing here teaches measurement; it is not a benchmark claim.

Production kernels add vectorized loads, larger/register tiles, simdgroup
instructions, half precision, and hardware-specific tuning.

## Files worth reading

Read:

1. `tiled_matmul.hpp` — the public API and fixed tile constants.
2. The Metal kernel in `tiled_matmul.cpp`.
3. The two barrier comments inside the tile loop.
4. The `[9,10] × [10,11]` comparison in `demo.cpp`.

You can skim the printing, timing helper, and repeated error checks.

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
  outputs/infer-lab/phase0/lesson30-tiled-matmul/tiled_matmul.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson30-tiled-matmul/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson30_tiled_matmul

./work/lesson30_tiled_matmul
```

## Main takeaways

1. One 64-lane group computes one 8×8 output tile.
2. Threadgroup memory lets neighboring lanes reuse input values.
3. The first barrier separates tile writes from tile reads.
4. The second barrier separates tile reads from the next overwrite.
5. All lanes must reach both barriers unconditionally.
6. Invalid edge loads become zero instead of returning early.
7. Only valid output lanes store results.
8. Correctness is checked against both CPU and the naive GPU baseline.
9. Timing results are observations, not correctness requirements.
