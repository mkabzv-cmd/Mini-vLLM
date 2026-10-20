# Lesson 10: CPU-backed MiniGPU runtime

This lesson builds the first real MiniGPU subsystem. It does not use the
Apple GPU yet. Instead, one background CPU thread executes queued commands
while preserving the API and synchronization rules that a Metal backend
will later implement.

## Objects

- `Device` creates buffers and queues.
- `Buffer` owns FP32 storage.
- `Kernel` describes what one logical GPU thread does.
- `LaunchConfig` describes groups and threads per group.
- `Queue` accepts commands in FIFO order and executes them asynchronously.
- `Event` reports eventual completion or failure.

The key separation is:

```text
caller → MiniGPU API → CPU backend now
                     → Metal backend later
```

Upper layers such as MiniTensor and Mini-vLLM should not need to change when
the backend changes.

## Build and run

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

Expected result:

```text
[PASS] vector add and padded threads
[PASS] delayed out-of-bounds error
[PASS] FIFO queue ordering
[PASS] pending command retains buffers
[PASS] queue recovery after failure
All MiniGPU CPU tests passed
```

## What the tests prove

1. Five work items can be covered by eight padded logical threads safely.
2. Kernel execution errors appear later through `Event::wait()`.
3. Commands in one queue execute in submission order.
4. Queued kernels retain buffer storage even after caller handles disappear.
5. One failed command does not permanently destroy the queue.

## Deliberately missing

- Real GPU execution or timing.
- SIMD groups and divergence.
- Threadgroup memory and barriers.
- Atomics.
- Multiple queues and dependencies.
- A kernel language or compiler.
- Metal.

The purpose of this version is to establish correct contracts for ownership,
ordering, asynchronous completion, and error propagation.
