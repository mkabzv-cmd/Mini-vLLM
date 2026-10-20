# Lesson 14 — A reusable MiniGPU Metal layer

## Outcome

Lesson 13 placed every Metal operation directly inside one program.
Lesson 14 moves those operations behind a small reusable API:

```text
demo.mm
   ↓ uses
minigpu.hpp
   ↓ implemented by
minigpu.mm
   ↓ calls
Apple Metal
   ↓
Apple GPU
```

The demo still calculates:

```text
[1, 2, 3, 4, 5]
+
[10, 20, 30, 40, 50]
        ↓ real Apple GPU
[11, 22, 33, 44, 55]
```

But `demo.mm` no longer creates Metal buffers, command buffers, encoders,
or compute pipelines itself.

## The public API

```cpp
minigpu::Device device;
auto queue = device.create_queue();

minigpu::Buffer a = device.allocate(5);
minigpu::Buffer b = device.allocate(5);
minigpu::Buffer output = device.allocate(5);

minigpu::Kernel vector_add =
    device.compile_kernel(
        "vector_add",
        vector_add_source
    );

minigpu::Dispatch dispatch(vector_add);
dispatch
    .bind_buffer(0, a)
    .bind_buffer(1, b)
    .bind_buffer(2, output)
    .bind_u32(3, 5);

minigpu::Event finished = queue->submit(
    std::move(dispatch),
    minigpu::LaunchConfig::cover(5, 4)
);

finished.wait();
```

This is now recognizably our own small accelerator runtime.

## Architecture

```text
User code
   │
   ├── Device
   │     ├── allocate()       → MTLBuffer
   │     ├── compile_kernel() → MTLComputePipelineState
   │     └── create_queue()   → MTLCommandQueue
   │
   ├── Dispatch
   │     ├── chosen Kernel
   │     ├── buffer bindings
   │     └── scalar bindings
   │
   └── Queue::submit()
         ├── creates MTLCommandBuffer
         ├── creates compute encoder
         ├── records bindings and launch
         ├── commits asynchronously
         └── returns Event
```

The public header contains ordinary C++ types. Objective-C Metal objects
remain hidden inside `minigpu.mm`.

## What each object means

### `Device`

The connection to one Metal GPU and a factory for its resources:

```cpp
Device device;
Buffer buffer = device.allocate(100);
Kernel kernel = device.compile_kernel(name, source);
auto queue = device.create_queue();
```

### `Buffer`

A shared-memory `MTLBuffer` hidden behind a bounds-checked C++ object:

```cpp
buffer.store(3, 7.0f);
float value = buffer.load(3);
```

These CPU accesses are safe only when the GPU is not concurrently
modifying the buffer. Wait for the relevant `Event` first.

### `Kernel`

Compiled GPU code:

```text
MSL source string
    ↓ Device::compile_kernel()
MTLLibrary
    ↓ select named function
MTLFunction
    ↓ build pipeline
Kernel containing MTLComputePipelineState
```

The kernel describes the algorithm but not one particular execution.

### `Dispatch`

One invocation of a kernel with concrete arguments:

```cpp
Dispatch dispatch(vector_add);
dispatch.bind_buffer(0, a);
dispatch.bind_buffer(1, b);
dispatch.bind_buffer(2, output);
dispatch.bind_u32(3, element_count);
```

This separation matters:

```text
Kernel
    reusable compiled program

Dispatch
    one use of that program with particular buffers and values
```

`Dispatch` is move-only because submission consumes that one invocation
description:

```cpp
queue->submit(std::move(dispatch), launch);
```

Moving the `Dispatch` transfers its kernel and argument handles into the
submission. It does not move or copy the underlying GPU buffer contents.

The binding indexes must still match the MSL declarations:

```text
bind_buffer(0, a)      ↔ [[buffer(0)]]
bind_buffer(1, b)      ↔ [[buffer(1)]]
bind_buffer(2, output) ↔ [[buffer(2)]]
bind_u32(3, count)     ↔ [[buffer(3)]]
```

### `LaunchConfig`

Describes the number of groups and threads:

```cpp
LaunchConfig::cover(5, 4)
```

becomes:

```text
2 groups × 4 threads = 8 launched threads
```

### `Queue`

Turns a `Dispatch` into real Metal commands and commits them:

```text
Dispatch
   ↓ Queue::submit()
MTLCommandBuffer
   ↓ encoder records pipeline, arguments, launch
commit()
```

### `Event`

Wraps completion of the Metal command buffer:

```cpp
Event finished = queue->submit(...);
finished.wait();
```

`wait()` calls Metal's `waitUntilCompleted()` and reports a GPU error by
throwing `MetalError`.

In the CPU simulator, we implemented event completion ourselves with a
condition variable. Metal already tracks command-buffer completion, so
this backend can wrap Metal's existing mechanism.

## The most important new boundary

Lesson 13 mixed three responsibilities:

```text
application logic
Metal runtime logic
GPU kernel source
```

Lesson 14 separates them:

```text
demo.mm       application logic
minigpu.mm    Metal runtime logic
MSL string    GPU algorithm
```

This prepares for our compiler:

```text
Future compiler
    ↓ generates MSL source
Device::compile_kernel()
    ↓ returns reusable Kernel
Dispatch
    ↓ binds tensors and scalar values
Queue
    ↓ runs it on the Apple GPU
```

The future compiler will not need to understand Objective-C Metal command
encoding. It will target this smaller API.

## Buffer lifetime

`Buffer` holds shared ownership of its hidden Metal storage. Binding a
buffer copies that handle into the `Dispatch`.

When submitted, the returned `Event` retains the kernel and its bound
buffers:

```text
caller Buffer ─────┐
Dispatch/Event ────┼── hidden MTLBuffer
Metal command ─────┘
```

Destroying one caller handle therefore does not immediately destroy the
underlying storage needed by that submitted command.

This solves storage lifetime. It does not make concurrent CPU/GPU access
safe; execution synchronization still requires an `Event`.

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
    minigpu.mm \
    demo.mm \
    -framework Foundation \
    -framework Metal \
    -o lesson14

./lesson14
```

Expected output:

```text
Metal device: Apple ...
Launch: 2 groups x 4 threads = 8 launched threads
Output: [11, 22, 33, 44, 55]
PASS: reusable MiniGPU API ran vector_add on the real GPU.
```

## Main takeaways

1. `minigpu.hpp` is our API; `minigpu.mm` is its Metal implementation.
2. Metal-specific Objective-C types are hidden from API users.
3. A `Kernel` is compiled code; a `Dispatch` is one invocation.
4. Bindings connect host buffers and values to kernel argument slots.
5. `Queue::submit()` records and commits a real Metal command.
6. `Event` represents that command's eventual completion.
7. Buffer ownership and execution synchronization are separate concerns.
8. The API is now a target for our future kernel compiler.

## What to skim

You may skim:

- Objective-C reference-management details
- `Impl` structs used to hide Metal types
- repetitive `nil` and bounds checks
- conversion of `NSError` into `MetalError`

Focus on:

```text
compile Kernel
→ construct Dispatch
→ bind arguments
→ submit through Queue
→ wait on Event
→ read Buffer
```

## Deliberately missing

- Pipeline caching
- A separate offline `.metal` build
- More scalar types and multidimensional launches
- Threadgroup shared-memory configuration
- Multiple kernels inside one command buffer
- Nonblocking callbacks and dependency graphs
- Real Metal reduction, atomics, and profiling
- Tensor shapes, strides, and dtypes

Those later features should be added only when an upper layer needs them.
