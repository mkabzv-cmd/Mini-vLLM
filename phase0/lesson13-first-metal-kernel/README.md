# Lesson 13 — First real Apple GPU kernel

## Outcome

This lesson performs:

```text
[1, 2, 3, 4, 5]
+
[10, 20, 30, 40, 50]
        ↓ real Apple GPU
[11, 22, 33, 44, 55]
```

Lessons 10–12 simulated GPU ideas with CPU threads. This lesson uses
Apple's Metal API to execute a Metal kernel on the physical Apple GPU.

It deliberately remains one small program. The reusable Metal
implementation of our `Device`, `Buffer`, `Queue`, `Kernel`, and `Event`
abstractions belongs in Lesson 14.

## The architecture

```text
main.mm host code                         vector_add kernel
runs on the CPU                           runs on the GPU

create Metal device
compile kernel source ------------------> executable GPU pipeline
allocate and fill buffers --------------> GPU reads a and b
record vector-add command --------------> GPU writes output
commit command
wait for completion <-------------------- command finishes
read and verify output
```

Metal is the Apple-provided API, compiler runtime, and driver interface
between our program and the Apple GPU:

```text
Our Objective-C++ program
        ↓
Metal framework and driver
        ↓
Physical Apple GPU
```

We are writing the algorithm and controlling its buffers and launch.
Apple supplies the low-level driver, GPU scheduling, and physical
hardware.

## Mapping from MiniGPU to Metal

| MiniGPU idea | Real Metal object |
|---|---|
| `Device` | `MTLDevice` |
| Data `Buffer` | `MTLBuffer` |
| `Kernel` | Metal function plus `MTLComputePipelineState` |
| `Queue` | `MTLCommandQueue` |
| One submitted batch | `MTLCommandBuffer` |
| Command recorder | `MTLComputeCommandEncoder` |
| `LaunchConfig` | threadgroups and threads per threadgroup |
| `Event` | later wrapper around command-buffer completion |

Two similarly named objects have very different jobs:

```text
MTLBuffer
    stores input or output data

MTLCommandBuffer
    stores recorded commands that will be submitted to the GPU
```

## The GPU kernel

The kernel is embedded in `main.mm` as Metal Shading Language source:

```cpp
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    output[global_id] = a[global_id] + b[global_id];
}
```

Each GPU thread receives a different `global_id`:

```text
thread 0: output[0] = a[0] + b[0]
thread 1: output[1] = a[1] + b[1]
thread 2: output[2] = a[2] + b[2]
...
```

There is no barrier because threads do not cooperate. Each thread reads
two elements and writes one separate output element.

## Launch geometry

The program asks for four threads per group while processing five
elements:

```text
group_count = ceil(5 / 4) = 2

Group 0: global IDs 0, 1, 2, 3
Group 1: global IDs 4, 5, 6, 7
```

Eight threads are launched:

```text
IDs 0–4: process five real elements
IDs 5–7: fail the bounds check and return
```

This is the real Metal equivalent of:

```cpp
LaunchConfig::cover(5, 4);
```

## Buffer bindings

The host and kernel agree through numbered slots:

```text
CPU host code                            GPU kernel declaration

setBuffer(a, index 0)       <--------->  [[buffer(0)]]
setBuffer(b, index 1)       <--------->  [[buffer(1)]]
setBuffer(output, index 2)  <--------->  [[buffer(2)]]
setBytes(count, index 3)    <--------->  [[buffer(3)]]
```

The numbers must match. Metal uses them to connect host resources to
kernel parameters.

## Why `.mm`?

Metal's host API uses Objective-C objects such as:

```text
id<MTLDevice>
id<MTLBuffer>
id<MTLCommandQueue>
```

The `.mm` extension tells the compiler that the file is Objective-C++.
It can therefore contain both normal C++ and Objective-C Metal calls.

The kernel itself is a separate language: Metal Shading Language, or
MSL. In this lesson it is stored as a string and compiled at runtime.

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
    main.mm \
    -framework Foundation \
    -framework Metal \
    -o lesson13

./lesson13
```

Expected structure of the output:

```text
Metal device: Apple ...
Elements: 5
Groups: 2
Threads per group: 4
Total launched threads: 8
a      = [1, 2, 3, 4, 5]
b      = [10, 20, 30, 40, 50]
output = [11, 22, 33, 44, 55]
PASS: vector_add executed correctly on the GPU.
```

The executable verifies every output value and exits with a nonzero
status if compilation, allocation, execution, or arithmetic fails.

## The execution timeline

```text
CPU                                      Apple GPU

create device and command queue
create three shared MTLBuffers
write a and b
compile vector_add
create command buffer
create compute encoder
bind pipeline and buffers
record threadgroup dispatch
end encoding
commit -------------------------------> queued for execution
waitUntilCompleted()                    run vector_add threads
                     <----------------- finish command
read output
```

`commit()` submits the command asynchronously. The CPU must wait before
reading the output. Lesson 14 will place this completion behavior behind
our own `Event` abstraction.

## What is important

Retain these ideas:

1. Host code runs on the CPU; Metal kernel code runs on the GPU.
2. An `MTLBuffer` stores data accessible to the GPU.
3. A compute pipeline is the executable form of a compiled kernel.
4. A compute encoder records the kernel, its arguments, and its launch.
5. Binding indexes on the CPU must match kernel `[[buffer(n)]]` indexes.
6. `commit()` submits asynchronous work.
7. The CPU waits before reading data the GPU is writing.
8. Padded threads require a bounds check.
9. Independent vector-add threads need no barrier.

## What is deliberately missing

- Reusable MiniGPU wrappers around Metal
- Separate offline `.metal` compilation
- Multiple kernels in one command buffer
- Asynchronous completion handlers
- Threadgroup shared memory and barriers on the real GPU
- Reductions, atomics, profiling, and optimization
- Our future kernel language and compiler

Those are later steps. This lesson proves only the first essential
vertical slice:

```text
CPU program → Metal → real Apple GPU → correct output
```
