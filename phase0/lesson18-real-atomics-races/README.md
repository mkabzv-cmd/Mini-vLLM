# Lesson 18 — Real GPU atomics and race experiments

## Outcome

This lesson runs three experiments on the real Apple GPU:

```text
A. ordinary load + add + store
   → data race, no guaranteed answer

B. atomic load + atomic store
   → no data race, but still loses updates

C. atomic fetch_add
   → no lost updates and unique tickets
```

This is the real-Metal version of Lesson 12's central ideas.

## Experiment A: ordinary competing updates

Many GPU threads execute:

```metal
const uint old_value = counter[0];
counter[0] = old_value + 1u;
```

One apparent increment is actually three steps:

```text
read the old value
add one privately
write the replacement value
```

Different threads may overlap:

```text
thread A reads 20
thread B reads 20
thread A writes 21
thread B writes 21
```

Two increment attempts produced only one increase.

This is a data race because different threads access the same ordinary
location, at least one writes, and no synchronization or atomic operation
orders those accesses.

The demo repeats the experiment and prints the observed values. It does not
assert that any particular racy result is correct:

```text
data-racy program → no guaranteed answer
```

Even if a racy kernel happens to print the requested count on one run, it is
still incorrect.

## Experiment B: atomic load plus atomic store

The second experiment uses one threadgroup atomic counter:

```metal
threadgroup atomic_uint counter;
```

Every individual access is atomic:

```metal
const uint old_value = atomic_load_explicit(
    &counter,
    memory_order_relaxed
);

atomic_store_explicit(
    &counter,
    old_value + 1u,
    memory_order_relaxed
);
```

However, the entire sequence is not one indivisible operation:

```text
atomic load
ordinary private addition
atomic store
```

The kernel deliberately places a barrier between the load and store:

```metal
old_value = atomic_load(...);

threadgroup_barrier(...);

atomic_store(..., old_value + 1);
```

With 64 threads:

```text
all 64 atomically load 0
barrier
all 64 atomically store 1
barrier
final counter = 1
```

There is no data race because every counter access is atomic. There is still
a race condition because the multi-step increment algorithm is wrong.

The barrier does not repair the algorithm. It forces the lost-update schedule
to happen every time.

## Experiment C: atomic read-modify-write

The correct operation is:

```metal
const uint ticket = atomic_fetch_add_explicit(
    counter,
    1u,
    memory_order_relaxed
);
```

This makes the complete sequence indivisible:

```text
read old value
add one
write new value
```

No other atomic operation on that location can appear inside the update.

For four threads, one possible order is:

```text
thread 2 receives 0, counter becomes 1
thread 0 receives 1, counter becomes 2
thread 3 receives 2, counter becomes 3
thread 1 receives 3, counter becomes 4
```

The thread order is not guaranteed. The important guarantees are:

```text
final counter = 4
returned tickets are uniquely 0, 1, 2, 3
```

The demo performs 1,000 logical increments across four threadgroups:

```text
1,000 logical threads
256 threads per group
4 groups
1,024 launched threads
24 padded threads
```

Padded threads return before performing the increment. The verified results
are:

```text
final counter = 1,000
sorted tickets = 0 through 999
```

## Why `fetch_add` returns the old value

Suppose the counter currently contains 37:

```metal
uint ticket = atomic_fetch_add_explicit(
    counter,
    1u,
    memory_order_relaxed
);
```

Afterward:

```text
ticket  = 37
counter = 38
```

This lets competing threads reserve unique positions:

```metal
uint slot = atomic_fetch_add_explicit(
    next_free_slot,
    1u,
    memory_order_relaxed
);

output[slot] = my_value;
```

This pattern later appears in compaction, token routing, work queues and
KV-cache page allocation.

## Threadgroup atomic versus device atomic

Experiment B declares:

```metal
threadgroup atomic_uint counter;
```

That means:

```text
one temporary counter per threadgroup
only threads in that group can access it
the counter disappears when the group finishes
```

Experiment C receives:

```metal
device atomic_uint* counter [[buffer(0)]];
```

That means:

```text
the counter lives in a bound Metal buffer
threads from every group can update it
it survives after individual groups finish
```

The address space determines which participants can reach the counter.

## Atomic is not a barrier

This is still insufficient:

```metal
atomic_fetch_add_explicit(
    counter,
    1u,
    memory_order_relaxed
);

if (local_id == 0) {
    use(atomic_load_explicit(counter, ...));
}
```

Every update is individually safe, but local thread zero may read before its
peers have performed their updates.

If one group needs the completed total:

```metal
atomic_fetch_add_explicit(...);

threadgroup_barrier(mem_flags::mem_threadgroup);

if (local_id == 0) {
    use(atomic_load_explicit(...));
}
```

Remember:

```text
atomic operation
    makes one update indivisible

barrier
    waits for every thread in one group
```

A group barrier still cannot wait for other groups.

## What `memory_order_relaxed` means here

This lesson needs only the atomic counter itself to be correct.

```metal
memory_order_relaxed
```

means the atomic operation remains indivisible, but it is not being used to
publish or order surrounding ordinary memory accesses.

It does not mean:

```text
non-atomic
threadgroup-only
unsafe counter
```

Acquire/release publication protocols are deliberately deferred.

## Why the counter needs a small readback kernel

Lesson 14's public `Buffer::load()` reads FP32 values, while Metal's atomic
counter is an unsigned integer.

Metal buffers themselves are untyped byte storage. For this lesson:

```text
one four-byte buffer slot
    ↓ viewed in MSL as
uint or atomic_uint
```

The CPU initializes the four bytes to zero. After an experiment, a one-thread
GPU kernel converts the integer counter to a float result that the existing
MiniGPU API can read:

```metal
result[0] = float(counter[0]);
```

This is temporary teaching plumbing. A future typed tensor/buffer layer will
represent U32 storage directly.

## Why the racy experiment is not a correctness test

Correct code has an expected answer:

```text
atomic fetch_add 1,000 times → exactly 1,000
```

Incorrect concurrent code has no valid expected answer:

```text
ordinary competing increment → unspecified observation
```

Therefore the program:

- prints the racy observations,
- does not require them to differ,
- does not call any observed racy value correct,
- tests only guaranteed atomic behavior.

This is how unsafe concurrency experiments should be interpreted.

## Connection to Lesson 17

Lesson 17:

```text
threads cooperate through several threadgroup-memory indexes
barriers separate reduction phases
```

Lesson 18:

```text
threads compete to update one memory location
atomics prevent lost updates
```

Later we can combine them:

```text
each group aggregates locally
        ↓
one group leader performs a device atomic update
```

That reduces global contention.

## Relevance to inference engineering

Atomics appear in:

```text
KV-cache page reference counts
free-list operations
token compaction
Mixture-of-Experts slot reservation
work queues
scheduler counters
histograms
sparse scatter updates
```

Correctness is only the first question. Thousands of threads updating one
hot atomic counter may become a performance bottleneck, which motivates
hierarchical aggregation and reductions.

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
    -o lesson18

./lesson18
```

The exact values printed by Experiment A may change. Experiments B and C
must remain:

```text
atomic load/store forced result = 1
atomic fetch_add final result   = 1,000
sorted tickets                  = 0 through 999
```

## Main takeaways

1. Ordinary same-location updates can create a data race.
2. A racy result has no guaranteed correct value.
3. Atomic load and atomic store are safe individual operations.
4. Separate atomic load and store calls do not create an atomic increment.
5. A barrier can force a lost-update schedule without fixing it.
6. `atomic_fetch_add` performs one indivisible read-modify-write.
7. `fetch_add` returns the old value, producing unique tickets.
8. `threadgroup atomic_uint` creates one counter per group.
9. `device atomic_uint*` can be shared across groups.
10. Atomics do not wait for other threads.
11. `memory_order_relaxed` preserves atomicity but does not publish unrelated
    ordinary memory.
12. Correct atomics may still serialize heavily contended updates.

## What to understand versus skim

Understand:

```text
ordinary load/add/store
atomic load/store pair
atomic fetch_add
old value / unique ticket
threadgroup atomic
device atomic
atomic versus barrier
```

Skim:

```text
readback conversion kernel
repeated racy trials
buffer-zero initialization detail
MSL function compilation boilerplate
output formatting
```

## Next

Lesson 19 will use real reductions and timing to compare:

```text
hot global atomic
threadgroup aggregation
multi-pass reduction
softmax building blocks
```
