# Lesson 19 — Reductions, softmax and performance measurement

## Outcome

This lesson combines the real-GPU concepts from Lessons 15, 17 and 18:

```text
launch geometry
threadgroup scratch memory
barriers
padded threads
multiple kernel passes
performance measurement
```

It performs three experiments on the Apple GPU:

```text
A. Complete multi-pass reduction
   10 values → 3 partials → 1 final sum

B. Numerically stable softmax
   4,096 large logits → 4,096 probabilities

C. End-to-end timing
   one serial GPU thread versus parallel multi-pass reduction
```

Lesson 14's MiniGPU runtime is reused unchanged.

## Part A: complete multi-pass reduction

Lesson 17 stopped after producing:

```text
[1, 2, 3, 4]   → 10
[5, 6, 7, 8]   → 26
[9, 10, 0, 0]  → 19

partial_sums = [10, 26, 19]
```

Lesson 19 launches the same reduction operation again:

```text
[10, 26, 19, 0] → 55
```

The complete shape is:

```text
10 → 3 → 1
```

The CPU reads only the final one-element result. It does not add the partials.

## One reduction pass

The kernel begins with:

```metal
threadgroup float scratch[256];
```

This is the maximum scratch capacity supported by this lesson. The actual
number of active scratch slots is determined by the launch's group width.

Every thread loads one input value:

```metal
scratch[local_id] =
    global_id < element_count
        ? input[global_id]
        : 0.0f;
```

Padded threads contribute zero and continue through every barrier.

After the first barrier, the active number of values is repeatedly halved:

```metal
for (
    uint stride = group_width / 2;
    stride > 0;
    stride >>= 1
) {
    if (local_id < stride) {
        scratch[local_id] +=
            scratch[local_id + stride];
    }

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );
}
```

For a four-thread group:

```text
stride 2:
[a, b, c, d]
 ↓  ↓
[a+c, b+d, c, d]

barrier

stride 1:
[a+c, b+d, c, d]
 ↓
[a+b+c+d, ...]
```

The group leader writes one partial:

```metal
if (local_id == 0) {
    output[group_id] = scratch[0];
}
```

## Why repeated kernel passes are necessary

A barrier synchronizes only one group.

Group zero cannot use a normal barrier to wait for groups one and two.
Therefore, one kernel produces independent group results:

```text
pass 1 output = [10, 26, 19]
```

The first dispatch then ends. A second dispatch consumes that buffer:

```text
pass 2 output = [55]
```

The dispatch boundary, together with submitting both dispatches to the same
ordered queue, provides global separation:

```text
all groups in pass 1 finish
        ↓
pass 2 consumes completed partials
```

Production inference kernels frequently use this pattern when a result
cannot be completed inside one threadgroup.

## Reduction workspace

The host calculates the output size of every pass before execution:

```text
1,048,576 inputs
        ↓
4,096 partials
        ↓
16 partials
        ↓
1 result
```

Buffers are allocated before timing:

```text
workspace output 0: 4,096 floats
workspace output 1: 16 floats
workspace output 2: 1 float
```

Each pass changes:

```text
input buffer
output buffer
logical element count
```

The compiled reduction kernel is reused.

The passes are submitted to the same queue in dependency order. The final
event is waited on, and the completed per-pass events are checked for errors.

## Part B: softmax

Softmax converts logits into probabilities:

```text
probability[i] =
    exp(logit[i]) /
    sum(exp(all logits))
```

The results are:

```text
nonnegative
sum approximately to one
larger logits receive larger probabilities
```

## Why naive softmax is numerically dangerous

The lesson uses logits around:

```text
996 through 1004
```

Calculating:

```text
exp(1004)
```

would overflow normal FP32 range.

Softmax is unchanged if the same constant is subtracted from every logit.
We subtract the maximum:

```text
shifted[i] = logit[i] - maximum
```

Then:

```text
largest shifted value = 0
largest exponential   = exp(0) = 1
```

Every other shifted value is negative, so its exponential is at most one.

The stable formula is:

```text
probability[i] =
    exp(logit[i] - maximum) /
    sum(exp(logit[j] - maximum))
```

## Softmax execution phases

This first softmax uses exactly one 256-thread group for one vector.

For 4,096 values, each thread handles several indexes:

```text
thread 0:
    indexes 0, 256, 512, ...

thread 1:
    indexes 1, 257, 513, ...

thread 255:
    indexes 255, 511, 767, ...
```

### Phase 1: lane-local maximum

Each thread finds the maximum among its assigned values:

```metal
float lane_maximum = very_negative_value;

for (
    uint index = local_id;
    index < element_count;
    index += group_width
) {
    lane_maximum =
        max(lane_maximum, input[index]);
}
```

### Phase 2: group maximum

The 256 lane-local maxima are placed into threadgroup memory and reduced:

```text
256 lane maxima
→ 128
→ 64
→ 32
→ ...
→ 1 vector maximum
```

A barrier separates every reduction level.

### Phase 3: shifted exponentials

Every thread processes its assigned values:

```metal
const float unnormalized =
    exp(input[index] - maximum);

output[index] = unnormalized;
lane_sum += unnormalized;
```

Each thread produces one private partial sum.

### Phase 4: denominator

The 256 lane-local sums are reduced through a second threadgroup scratch
array:

```text
256 partial sums → one denominator
```

### Phase 5: normalization

Every thread revisits its assigned output indexes:

```metal
output[index] *= 1.0f / denominator;
```

The GPU results are compared against a double-precision CPU reference.

The tests verify:

```text
every probability is finite
every probability is nonnegative
probability sum is approximately one
maximum CPU/GPU absolute error is small
```

## Softmax limitation

This kernel uses one threadgroup for one vector.

Advantages:

```text
one kernel dispatch
barriers can coordinate every participating thread
no device-wide intermediate buffers
```

Limitation:

```text
one vector occupies only one threadgroup
```

A future tensor layer can launch one group per row when processing many
softmax rows. Very large single rows may require multi-pass kernels or more
advanced SIMD-group techniques.

This lesson prioritizes understandable synchronization over a production
softmax implementation.

## Part C: performance experiment

Two GPU algorithms sum 1,048,576 values containing `1.0f`.

### Serial GPU baseline

One GPU thread loops through the complete buffer:

```metal
float total = 0.0f;

for (uint index = 0; index < element_count; ++index) {
    total += input[index];
}
```

This is a serial GPU implementation, not a CPU comparison.

### Parallel multi-pass reduction

The parallel version uses:

```text
1,048,576 values
256 threads per group
4,096 groups in pass 1

1,048,576 → 4,096 → 16 → 1
```

Every group reduces its values concurrently, then later passes reduce the
partials.

Both algorithms must return:

```text
1,048,576
```

The benchmark never accepts speed instead of correctness.

## What the timer measures

The lesson uses:

```cpp
std::chrono::steady_clock
```

around each complete operation:

```text
construct and encode Dispatch
submit command buffer
queue scheduling
GPU execution
Event::wait()
```

Therefore the result is end-to-end latency for our current MiniGPU runtime.
It is not pure GPU-kernel time.

Excluded from timing:

```text
MSL compilation
Metal pipeline creation
buffer allocation
input initialization
correctness checking
printing
```

The experiment performs:

```text
5 warm-up runs
21 measured samples
median latency
alternating serial/parallel execution order
```

Median is used because an occasional operating-system interruption can
distort one sample.

## Example result from the development Mac

One Apple M4 Pro run produced approximately:

```text
serial GPU sum           62.4 ms
parallel multi-pass       0.30 ms
serial/parallel ratio   210×
one-group softmax         0.12 ms
```

These values are not portable promises. They can change with:

```text
device
OS and driver
temperature and power state
background work
compiler version
input size
group size
measurement method
```

The reliable lesson is structural:

```text
one lane performs N operations serially
versus
many groups perform the work in parallel
```

For very small inputs, the parallel version can lose because extra launches
and synchronization cost more than the saved arithmetic.

## Build and run

From this lesson directory:

```bash
xcrun clang++ \
    -std=c++20 \
    -O2 \
    -g \
    -Wall \
    -Wextra \
    -Wpedantic \
    -fobjc-arc \
    ../lesson14-metal-minigpu/minigpu.mm \
    demo.mm \
    -framework Foundation \
    -framework Metal \
    -o lesson19

./lesson19
```

Expected structure:

```text
Part A: complete multi-pass GPU reduction
Reduction shape: 10 -> 3 -> 1
GPU result: 55

Part B: numerically stable softmax
Probability sum: approximately 1
Maximum CPU/GPU absolute error: small

Part C: end-to-end latency experiment
Reduction shape: 1048576 -> 4096 -> 16 -> 1
machine-dependent timing values

PASS
```

## Main takeaways

1. One reduction pass produces one partial result per group.
2. Repeated GPU passes reduce partials to one final scalar.
3. Ordered dispatches on the same queue separate global reduction passes.
4. Reduction workspace can be allocated before execution and reused.
5. Softmax contains maximum reduction, exponentiation, sum reduction and
   normalization.
6. Subtracting the maximum prevents exponential overflow.
7. Threads can process multiple elements by advancing in group-width steps.
8. Group barriers coordinate the phases of one softmax row.
9. Performance must be measured rather than inferred from thread count.
10. Compilation, allocation and warm-up should normally be excluded from
    steady-state kernel timing.
11. End-to-end latency and pure kernel time are different measurements.
12. Correctness must be verified alongside performance.

## What to understand versus skim

Understand:

```text
N → partials → one result
ordered same-queue dispatches between global reduction passes
threadgroup reduction tree
stable softmax formula
max reduction
sum reduction
lane-strided work
warm-up and median timing
what the timer includes
```

Skim:

```text
workspace validation checks
integer overflow checks
event-vector management
CPU reference implementation details
sample sorting
output formatting
Objective-C++ setup
```

## Where the roadmap goes next

Lesson 19 completes the compute-and-synchronization part of the real-GPU
block:

```text
launch geometry
real threadgroup memory
real barriers
real atomics
real reductions
stable softmax
basic performance measurement
```

Lesson 16 was intentionally left untouched earlier, so it still remains:

```text
unified versus private GPU memory
CPU/GPU visibility
explicit copies
synchronization around memory access
```

After Lesson 16 completes the real-GPU block, the next layer can move toward:

```text
compiler IR and Metal code generation
MiniTensor shapes, dtypes and operations
reusable optimized kernels
```
