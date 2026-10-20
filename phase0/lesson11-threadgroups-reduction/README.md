# Lesson 11: Threadgroup memory, barriers, and reduction

This lesson adds the first kind of cooperation between logical GPU threads.
Lesson 10 gave every thread an ID and let threads independently read and
write buffers. Now the threads inside one group can:

1. share temporary scratch memory,
2. wait for one another at reusable barriers, and
3. combine many values into one with a parallel reduction.

We still execute on the Mac CPU. This is a **functional simulator** for
learning correctness. It is not pretending that CPU threads have GPU
performance. The same concepts will later be mapped onto Metal and the
Apple GPU.

## Where this fits

```text
Apple GPU hardware                         reuse later
macOS GPU driver                          reuse later
Metal API + Metal shader compiler         reuse later
─────────────────────────────────────────────────────
MiniGPU API                               already started
├── buffers                               Lesson 10
├── asynchronous queue + events           Lesson 10
├── threadgroup memory                    this lesson
├── barriers                              this lesson
└── reduction kernel                      this lesson
─────────────────────────────────────────────────────
MiniTensor                                build later
TinyLLM                                   build later
Mini-vLLM                                 build later
Mini-Dynamo                               build later
```

The reduction may look small, but it is a direct ancestor of operations
needed by softmax, normalization, attention, sampling, and many tensor
kernels.

## 1. From independent threads to a cooperating group

Suppose four threads independently load four values:

```text
thread 0 loads 3
thread 1 loads 1
thread 2 loads 7
thread 3 loads 0
```

If every thread only has private variables, no single thread can see all
four values. We therefore give every threadgroup a small shared array:

```text
threadgroup 0
├── thread 0 ─┐
├── thread 1  ├── shared scratch memory
├── thread 2  │
└── thread 3 ─┘

threadgroup 1 ─── its own different scratch memory
```

Metal calls this **threadgroup memory**. CUDA calls the analogous idea
**shared memory**. The important concept is independent of either vendor.

Threadgroup memory has four defining properties:

- It is visible to threads in one threadgroup.
- It is not shared with other threadgroups.
- It is temporary: its lifetime is one group's execution.
- It is small scratch space intended for cooperation and data reuse.

Our `LaunchConfig` now includes the number of shared FP32 elements:

```cpp
minigpu::LaunchConfig(
    /* groups = */ 2,
    /* threads per group = */ 4,
    /* shared elements per group = */ 4
);
```

Each of the two groups receives a separate four-element shared array.

## 2. Why a barrier is necessary

Imagine that each thread runs this code:

```cpp
shared[local_id] = input[global_id];

if (local_id == 0) {
    use(shared[0], shared[1], shared[2], shared[3]);
}
```

The source code appears in that order **inside one thread**, but the four
threads do not necessarily advance together. One possible schedule is:

```text
thread 0 writes shared[0]
thread 0 immediately reads shared[1]
thread 1 has not written shared[1] yet
```

So source-code order inside thread 0 says nothing about how far thread 1
has progressed.

A barrier divides execution into phases:

```text
phase A: every thread writes its value
         ↓
barrier: nobody passes until every thread arrives
         ↓
phase B: threads may safely consume the completed values
```

A threadgroup barrier provides two related guarantees:

1. **Execution rendezvous:** every participating thread reaches the point
   before any participating thread continues.
2. **Memory visibility:** writes made before the barrier are visible to the
   group's reads after the barrier.

A barrier only coordinates one threadgroup. It is not a global barrier
across all groups in a normal kernel launch. Separate groups must be able to
run in any order, which is why large reductions use multiple kernel passes.

## 3. The barrier's most important rule

Every thread in the group must reach the same dynamic barriers in the same
order.

This is wrong:

```cpp
if (local_id < 4) {
    do_work();
    barrier();
}
```

Threads 0–3 wait at the barrier. Threads 4–7 skip it. The waiting threads
can never collect all eight arrivals.

This is correct:

```cpp
if (local_id < 4) {
    do_work();
}

barrier(); // every thread reaches it
```

The work may be conditional. The barrier must remain unconditional.

Early return creates the same problem:

```cpp
if (global_id >= element_count) {
    return; // dangerous when a later barrier exists
}

barrier();
```

In production GPU programming, divergent barrier behavior can hang or be
undefined. Our simulator deliberately converts common cases into
`DivergentBarrierError`, so a mistake becomes a readable test failure
instead of an endless wait.

Our teaching API writes:

```cpp
context.barrier(7);
```

The number is a **barrier site ID**. Real Metal and CUDA barriers do not
need this number. The simulator uses it to detect when one branch reaches
barrier site 7 while another branch reaches site 8. It is debugging
instrumentation for our miniature platform.

`ThreadContext` is valid only while the kernel callback is running. It
cannot be copied or moved. Never save its address or a reference for later;
its internal group state disappears after the callback finishes.

## 4. Reduction: turning many values into one

A reduction combines a collection with an operation:

```text
sum:      [3, 1, 7, 0] → 11
maximum:  [3, 1, 7, 0] → 7
product:  [3, 1, 7, 0] → 0
```

A sequential sum creates one long dependency chain:

```text
(((3 + 1) + 7) + 0) + ...
```

A tree reduction exposes independent work:

```text
[3, 1, 7, 0, 4, 1, 6, 3]
        stride 4
[7, 2, 13, 3]
        stride 2
[20, 5]
        stride 1
[25]
```

The complete shared-memory evolution is:

```text
load:
shared = [3, 1, 7, 0, 4, 1, 6, 3]
barrier 0

stride 4:
thread 0: shared[0] = 3 + 4 = 7
thread 1: shared[1] = 1 + 1 = 2
thread 2: shared[2] = 7 + 6 = 13
thread 3: shared[3] = 0 + 3 = 3
barrier 1

stride 2:
thread 0: shared[0] = 7 + 13 = 20
thread 1: shared[1] = 2 + 3 = 5
barrier 2

stride 1:
thread 0: shared[0] = 20 + 5 = 25
barrier 3

thread 0 writes shared[0] to the output buffer
```

There is a barrier after every stage because the next stage consumes values
produced by the previous stage.

The number of active arithmetic threads halves each time:

```text
8 inputs → 4 additions → 2 additions → 1 addition
```

All eight threads still reach every barrier, including threads that have no
addition to perform in later stages.

## 5. The kernel we built

Here is the essential structure, slightly shortened:

```cpp
float value =
    global_id < element_count
        ? input[global_id]
        : 0.0f;

shared[local_id] = value;
barrier(0);

for (stride = group_size / 2; stride > 0; stride /= 2) {
    if (local_id < stride) {
        shared[local_id] += shared[local_id + stride];
    }

    barrier(next_site);
}

if (local_id == 0) {
    group_results[group_id] = shared[0];
}
```

This first version requires a power-of-two group size such as 2, 4, 8, 16,
or 32. That keeps the tree regular. We will later generalize algorithms and
learn why real kernels care about SIMD-group width and hardware limits.

## 6. Padded threads and identity values

Suppose we have five values but launch groups of four:

```text
element count:       5
threadgroup size:    4
group count:         ceil(5 / 4) = 2
threads launched:    8

valid global IDs:    0 1 2 3 4
padded global IDs:             5 6 7
```

The padded threads cannot return because the kernel contains barriers.
Instead, they load the sum operation's **identity value**:

```text
x + 0 = x
```

So the final group effectively reduces:

```text
[input[4], 0, 0, 0]
```

Every padded lane then participates in every barrier.

Different operations need different identities:

```text
sum       0
product   1
maximum   negative infinity
minimum   positive infinity
```

This “do neutral work but keep participating” pattern appears constantly in
parallel kernels.

## 7. Why reduction needs multiple kernel passes

Threadgroup memory is private to a group. If ten values are reduced with
four threads per group, the first pass creates three independent partial
sums:

```text
input count       10
group size         4
group count        3

group 0 reduces input[0..3]   → partial[0]
group 1 reduces input[4..7]   → partial[1]
group 2 reduces input[8..9]   → partial[2]
```

No group may directly read another group's scratch memory. Therefore, a
second kernel launch reduces the three partials:

```text
10 values → 3 partial sums → 1 final sum
```

For a larger input the sequence might be:

```text
1000 → 125 → 16 → 2 → 1       with group size 8
```

The host keeps launching passes until only one element remains.

Our beginner `reduce_sum` waits after every pass. This is deliberate:

- FIFO order guarantees that pass 2 runs after pass 1.
- FIFO order alone does **not** say that pass 2 should be cancelled if pass
  1 failed.
- Waiting on each pass reports the exact earlier failure before a later
  pass consumes incomplete data.

Later, when we build dependency-aware command graphs, we can keep all
passes asynchronous while propagating failures through dependencies.

## 8. What changed inside the CPU simulator

Lesson 10 interpreted logical threads one at a time. That cannot model a
barrier: the first logical thread would wait forever before the interpreter
ever starts the second.

For this lesson only, each small simulated threadgroup works like this:

```text
queue worker
└── execute group 0
    ├── CPU thread representing lane 0
    ├── CPU thread representing lane 1
    ├── CPU thread representing lane 2
    └── CPU thread representing lane 3
        └── reusable barrier + shared scratch array

then execute group 1 with a fresh scratch array
```

Groups execute sequentially to keep resource use bounded. Threads within
one group execute concurrently. We cap a group at 64 simulated lanes
because creating one operating-system thread per GPU lane would be a
terrible production implementation.

A real GPU has hardware-managed execution lanes and can keep enormous
numbers of logical threads in flight efficiently. Our OS-thread mapping is
only a correctness microscope.

The simulator also contains:

- a start gate, so partial failure while creating CPU threads cannot strand
  a half-created group at a full-group barrier;
- a reusable condition-variable barrier;
- group cancellation, so an exception in one lane wakes waiting peers;
- bounds-checked global and shared memory;
- barrier-site validation;
- fresh threadgroup memory for each group;
- queue recovery after a failed kernel.

The simulator locks individual memory operations to avoid C++ data-race
undefined behavior when a learner writes a bad kernel. Those locks do not
make a logically racy GPU algorithm correct; they only let the simulator
report and study it safely. A later lesson will add explicit race-detection
instrumentation.

The same `Kernel` body is invoked concurrently by every lane in a group.
Captured `Buffer` handles are safe in this lesson because their operations
are synchronized. Any other captured mutable host object must also be
thread-safe. For example, incrementing an ordinary captured integer from
every lane would be a C++ data race and would not model valid GPU memory.

This simulator can cancel lanes that are waiting in `context.barrier()`.
It cannot forcibly stop arbitrary C++ code. A lane that enters an infinite
loop, waits forever on a host condition, or waits for a later event from
the same FIFO queue can deadlock the queue. Real runtimes also forbid or
carefully constrain circular command dependencies; our later dependency
graph will detect them before execution.

## 9. Connection to softmax

Stable softmax for logits `x` is:

```text
m = maximum(x)
e[i] = exp(x[i] - m)
s = sum(e)
y[i] = e[i] / s
```

It needs two reductions:

1. a maximum reduction,
2. a sum reduction.

Subtracting the maximum keeps exponentials numerically stable. The
threadgroup-memory and barrier rules from this lesson are therefore direct
building blocks for the softmax kernel we will implement later.

## 10. Connection to attention

For one query, simplified attention does:

```text
scores        = query × keys
probabilities = softmax(scores)
output        = probabilities × values
```

Optimized attention kernels cooperate to:

- load query, key, and value tiles,
- compute partial dot products,
- reduce those partial dot products,
- reduce maxima for stable softmax,
- reduce sums of exponentials,
- combine weighted values.

Eventually we will learn online softmax and FlashAttention-style tiling.
The vocabulary in those algorithms—tiles, shared memory, synchronization,
partial reductions, and numerical stability—starts here.

## 11. Build and run

From this directory:

```bash
clang++ \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -pthread \
  minigpu.cpp \
  tests.cpp \
  -o minigpu_tests

./minigpu_tests
```

Expected final line:

```text
All threadgroup and reduction tests passed
```

For memory and undefined-behavior checks:

```bash
clang++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  -pthread \
  minigpu.cpp \
  tests.cpp \
  -o minigpu_tests_sanitized

./minigpu_tests_sanitized
```

## 12. What the tests prove

1. Writes before a barrier are visible after it.
2. Every group receives isolated scratch memory.
3. Ten values reduce through `10 → 3 → 1`.
4. Padded lanes contribute zero and still reach barriers.
5. Repeated runs are independent of CPU-thread scheduling.
6. A shared-memory bounds error cancels the group without hanging.
7. The original error is delivered through `Event::wait()`.
8. A failed command does not destroy the queue.
9. Returning before a peer barrier becomes `DivergentBarrierError`.
10. Different barrier site IDs are rejected.
11. Reduction shapes that cannot work are rejected early.
12. Empty and one-element reductions obey their mathematical definitions.
13. The smallest valid group and the simulator's group-size cap are checked.
14. The same barrier object works across three separate generations.

## 13. Suggested reading order

Do not try to absorb the whole runtime at once:

1. Read the reduction lambda near `enqueue_reduce_sum_pass`.
2. Read the first three tests.
3. Read the public API in `minigpu.hpp`.
4. Read `ThreadContext`.
5. Read `ThreadgroupBarrier`.
6. Read `GroupExecutor`.
7. Read the asynchronous `Queue` last.

That order follows the concept you care about before its supporting
machinery.

## 14. Exercises

### Exercise A: predict before running

Manually reduce this with a group size of eight:

```text
[2, 5, 1, 8, 0, 3]
```

First add the two padded zeros, then write the shared array after each
stride.

### Exercise B: create the bug

Move the reduction barrier inside:

```cpp
if (local_id < stride) {
    ...
}
```

Run the tests and inspect the divergence error. Then restore it.

### Exercise C: maximum

Implement `reduce_max`. Remember that padded lanes need negative infinity,
not zero. Add tests containing only negative values; that catches the
common incorrect identity.

### Exercise D: one pass

Call `enqueue_reduce_sum_pass` directly for ten inputs with a group size of
four. Inspect the three partial sums before doing the final pass.

## Checkpoint questions

You understand this lesson when you can answer these without memorizing:

1. Who can access a group's threadgroup memory?
2. Why is threadgroup memory different from a global `Buffer`?
3. What two guarantees does a barrier provide?
4. Why must every lane reach the same barriers?
5. Why can the arithmetic be conditional while the barrier cannot?
6. What does a padded sum lane load, and why?
7. Why does a ten-value, four-thread reduction need two launches?
8. Why does waiting only for a later FIFO command not automatically report
   an earlier command's failure?
9. Which two reductions are required by stable softmax?
10. Why is one CPU thread per simulated GPU lane useful for teaching but
    unsuitable for production?

## Deliberately still missing

- Real Apple GPU execution through Metal.
- SIMD-group/warp-level execution.
- Atomics.
- Race detection.
- General non-power-of-two reduction.
- Maximum reduction and softmax.
- Kernel compilation.
- Multiple queues and dependency graphs.
- Performance benchmarking of real GPU kernels.

The next conceptual step is **atomics and race detection**, followed by the
first real Metal compute kernel. At that point the same MiniGPU concepts
will begin running on the Apple GPU rather than only in our CPU simulator.
