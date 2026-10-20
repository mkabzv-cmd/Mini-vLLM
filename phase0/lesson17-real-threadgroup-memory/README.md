# Lesson 17 — Real threadgroup memory and barriers

## Outcome

This lesson moves Lesson 11's most important idea from the CPU simulator onto
the real Apple GPU.

Ten values are divided among three four-thread groups:

```text
Group 0: [1,  2, 3, 4] → 10
Group 1: [5,  6, 7, 8] → 26
Group 2: [9, 10, 0, 0] → 19
                              ↑ padded threads contribute zero
```

The GPU writes:

```text
partial_sums = [10, 26, 19]
```

This is one real reduction pass. Lesson 19 will reduce partial results again
on the GPU and add performance measurement.

## The three relevant memory locations

Inside the kernel:

```metal
float value;
```

is private to one thread.

```metal
threadgroup float scratch[4];
```

creates one temporary four-float array for each threadgroup.

```metal
device float* partial_sums;
```

points to a persistent Metal buffer accessible through the device address
space.

The launch therefore looks like:

```text
Group 0 threads ──> Group 0 scratch[4] ──> partial_sums[0]
Group 1 threads ──> Group 1 scratch[4] ──> partial_sums[1]
Group 2 threads ──> Group 2 scratch[4] ──> partial_sums[2]
```

The three groups do not share the same `scratch` array. Metal creates a
separate instance for each group.

## Simulator-to-Metal mapping

```text
CPU simulator                         Real Metal

context.global_id()                   [[thread_position_in_grid]]
context.group_id()                    [[threadgroup_position_in_grid]]
context.local_id()                    [[thread_position_in_threadgroup]]
simulated group shared array          threadgroup float scratch[4]
context.barrier(7)                    threadgroup_barrier(...)
```

Real Metal barriers do not need our simulator's barrier-site number.

## Phase 1: load group scratch memory

Each thread uses its local ID as the shared-memory index:

```metal
scratch[local_id] =
    global_id < element_count
        ? input[global_id]
        : 0.0f;
```

For group zero:

```text
local 0 loads input[0] = 1 → scratch[0]
local 1 loads input[1] = 2 → scratch[1]
local 2 loads input[2] = 3 → scratch[2]
local 3 loads input[3] = 4 → scratch[3]

scratch = [1, 2, 3, 4]
```

For the final group:

```text
local 0 / global 8  → scratch[0] = 9
local 1 / global 9  → scratch[1] = 10
local 2 / global 10 → scratch[2] = 0
local 3 / global 11 → scratch[3] = 0

scratch = [9, 10, 0, 0]
```

Zero is the additive identity:

```text
x + 0 = x
```

so padded threads do not change the sum.

## First barrier

Writing different scratch indexes is safe, but one thread must not read
another thread's value before that write finishes.

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

This means:

```text
Every thread in this group must finish the preceding phase.
No thread in this group continues until all four arrive.
Earlier threadgroup-memory writes are visible afterward.
```

It does not wait for threads in other groups.

## Phase 2: combine pairs

Only local threads zero and one perform arithmetic:

```metal
if (local_id < 2) {
    scratch[local_id] += scratch[local_id + 2];
}
```

For group zero:

```text
local 0: scratch[0] = 1 + 3 = 4
local 1: scratch[1] = 2 + 4 = 6

scratch = [4, 6, 3, 4]
```

Local threads two and three do no arithmetic, but they still reach:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

The work is conditional. The barrier remains unconditional.

## Phase 3: produce one group result

After the second barrier, local thread zero can safely combine:

```metal
if (local_id == 0) {
    scratch[0] += scratch[1];
}
```

For group zero:

```text
scratch[0] = 4 + 6 = 10
```

Every thread reaches one final barrier. Then the group leader writes the
persistent result:

```metal
if (local_id == 0) {
    partial_sums[group_id] = scratch[0];
}
```

The mapping is:

```text
group 0 leader → partial_sums[0] = 10
group 1 leader → partial_sums[1] = 26
group 2 leader → partial_sums[2] = 19
```

## Why padded threads cannot return early

This would be wrong:

```metal
if (global_id >= element_count) {
    return;
}

threadgroup_barrier(...);
```

In group two:

```text
local threads 0 and 1 reach the barrier
local threads 2 and 3 already returned
```

The two waiting threads can never collect all four arrivals.

Instead, padded threads:

```text
write zero
perform no useful arithmetic
still reach every barrier
```

This is why a bounds check with early return was safe in Lesson 15's
independent elementwise kernel but is unsafe before a later barrier.

## Barrier versus Event

These operate at different levels:

```text
threadgroup_barrier()
    called by GPU threads inside a kernel
    waits for one threadgroup

Event::wait()
    called by the CPU
    waits for the entire submitted GPU command
```

A barrier cannot synchronize different groups.

To synchronize groups globally:

```text
finish kernel pass 1
wait or establish a queue dependency
launch kernel pass 2
```

That is how a future full reduction will consume `[10, 26, 19]`.

## Why the CPU sums the partials in this demo

After the GPU produces:

```text
[10, 26, 19]
```

the demo calculates:

```text
10 + 26 + 19 = 55
```

on the CPU only to verify correctness.

The CPU is not part of the reduction algorithm we ultimately want. Lesson 19
will launch another GPU pass.

## Important fixed-size assumption

The kernel declares:

```metal
threadgroup float scratch[4];
```

Therefore this first kernel must be launched with:

```text
threads_per_group = 4
```

The size is deliberately fixed so the memory behavior is visible.

Later, the runtime can support dynamically sized threadgroup memory or
compile specialized kernels for different group sizes.

## Build and run

From this lesson directory:

```bash
xcrun clang++ \
    -std=c++20 \
    -O0 \
    -g \
    -Wall \
    -Wextra \
    -Wpedantic \
    -fobjc-arc \
    ../lesson14-metal-minigpu/minigpu.mm \
    demo.mm \
    -framework Foundation \
    -framework Metal \
    -o lesson17

./lesson17
```

Expected:

```text
group 0 covers global IDs 0-3  -> partial sum 10
group 1 covers global IDs 4-7  -> partial sum 26
group 2 covers global IDs 8-11 -> partial sum 19

Partial sums: [10, 26, 19]
CPU verification total: 55
PASS: real threadgroup memory and barriers produced [10, 26, 19].
```

## Main takeaways

1. `threadgroup` memory is one temporary shared array per group.
2. Local ID selects a location inside the group's scratch array.
3. Different groups receive different scratch arrays.
4. A barrier waits only for threads in one group.
5. Every thread must reach the same barriers in the same order.
6. Conditional arithmetic is fine; a group barrier must remain
   unconditional.
7. Padded reduction threads contribute zero instead of returning.
8. Threadgroup memory disappears after the group finishes.
9. A group leader writes the result into a persistent device buffer.
10. Synchronization between groups requires a kernel boundary.

## What to understand versus skim

Understand:

```text
private value
threadgroup scratch memory
device output buffer
local ID
barrier phases
padded lanes contribute zero
one partial result per group
```

Skim:

```text
Objective-C++ setup
buffer initialization
output formatting
floating-point comparison helper
Metal error plumbing inherited from Lesson 14
```

## Next

Lesson 18 uses real Metal atomics to compare:

```text
ordinary competing update
atomic load followed by atomic store
atomic fetch_add
```
