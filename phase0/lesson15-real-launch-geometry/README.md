# Lesson 15 — Real GPU launch geometry

## Outcome

This lesson runs a kernel on the real Apple GPU and makes its launch geometry
visible.

We launch:

```text
logical tensor elements = 10
threads per group       = 4
groups                  = ceil(10 / 4) = 3
launched threads        = 3 × 4 = 12
padded threads          = 12 - 10 = 2
```

The GPU kernel records each thread's:

```text
global ID
group ID
local ID
active or padded status
```

It also doubles ten input values while ensuring that the two padded threads
do not access the ten-element input and output buffers.

This is the first lesson where IDs from the CPU simulator are replaced by
IDs supplied directly by Metal on the real GPU.

## The complete launch

One dispatch creates a grid of threads:

```text
Grid: 3 groups × 4 threads

Group 0
├── local 0 → global 0  → element 0
├── local 1 → global 1  → element 1
├── local 2 → global 2  → element 2
└── local 3 → global 3  → element 3

Group 1
├── local 0 → global 4  → element 4
├── local 1 → global 5  → element 5
├── local 2 → global 6  → element 6
└── local 3 → global 7  → element 7

Group 2
├── local 0 → global 8  → element 8
├── local 1 → global 9  → element 9
├── local 2 → global 10 → padded
└── local 3 → global 11 → padded
```

In a one-dimensional launch:

```text
global_id =
    group_id × threads_per_group + local_id
```

For example:

```text
group_id         = 2
local_id         = 1
threads_per_group = 4

global_id = 2 × 4 + 1 = 9
```

## Important vocabulary

### Grid or launch

Every thread created by one kernel dispatch:

```text
all 12 threads in this example
```

### Threadgroup

A fixed-size collection of threads:

```text
4 threads per group in this example
```

Threads inside a group will later be able to cooperate using threadgroup
memory and barriers.

### Local ID

The thread's position inside its own group:

```text
0, 1, 2, or 3
```

Every group starts its local IDs again from zero.

### Group ID

Which group contains the thread:

```text
0, 1, or 2
```

### Global ID

The thread's position across the entire grid:

```text
0 through 11
```

For simple elementwise kernels, the global ID commonly selects the tensor
element processed by the thread:

```metal
output[global_id] = input[global_id] * 2.0f;
```

The word `lane` in our simulator meant one logical kernel execution. Metal
normally calls it a thread. It is not a permanent physical GPU core.

## Why ceiling division is necessary

Ordinary integer division rounds down:

```text
10 / 4 = 2
```

Two groups would launch only eight threads and elements 8 and 9 would never
be processed.

We need enough complete groups to cover every element:

```text
ceil(10 / 4) = 3
```

`LaunchConfig::cover()` performs this calculation:

```cpp
const std::size_t groups =
    work_items == 0
        ? 0
        : 1 + (work_items - 1) / threads_per_group;
```

For ten elements and four threads per group:

```text
1 + (10 - 1) / 4
= 1 + 9 / 4
= 1 + 2
= 3 groups
```

## How Metal gives IDs to the kernel

The kernel declares special parameters:

```metal
uint global_id [[thread_position_in_grid]],
uint group_id [[threadgroup_position_in_grid]],
uint local_id [[thread_position_in_threadgroup]]
```

The values are not supplied through `Dispatch::bind_*()`. Metal fills them
for each thread when it executes the kernel.

Every thread executes the same kernel function, but receives different ID
values.

Conceptually:

```text
thread in group 0, local 0 receives:
    global_id = 0, group_id = 0, local_id = 0

thread in group 1, local 2 receives:
    global_id = 6, group_id = 1, local_id = 2

thread in group 2, local 3 receives:
    global_id = 11, group_id = 2, local_id = 3
```

This replaces the simulator's:

```cpp
context.global_id();
context.group_id();
context.local_id();
```

## Padded threads are real launched threads

Our tensor has valid indexes:

```text
0 through 9
```

But the launch also creates threads with:

```text
global_id = 10
global_id = 11
```

These threads are called padded threads because they exist only to complete
the final four-thread group.

Metal does not automatically know that they should avoid the tensor. The
kernel must check:

```metal
if (global_id >= element_count) {
    return;
}
```

Without this check, the padded threads could execute:

```metal
output[10] = input[10] * 2.0f;
output[11] = input[11] * 2.0f;
```

But ten-element buffers have no indexes 10 or 11. Those would be
out-of-bounds memory accesses.

The bounds check is therefore a correctness requirement, not merely a
performance optimization.

## Why some buffers contain 10 values and others contain 12

The program allocates:

```text
input and output buffers       10 elements
geometry diagnostic buffers   12 elements
```

The real tensor buffers contain only logical data:

```cpp
Buffer input = device.allocate(element_count);
Buffer output = device.allocate(element_count);
```

The diagnostic buffers intentionally contain a slot for every launched
thread:

```cpp
Buffer global_ids = device.allocate(launched_threads);
Buffer group_ids = device.allocate(launched_threads);
Buffer local_ids = device.allocate(launched_threads);
Buffer active_flags = device.allocate(launched_threads);
```

That lets padded threads safely record that they exist before returning:

```metal
global_ids[global_id] = float(global_id);
active_flags[global_id] =
    global_id < element_count ? 1.0f : 0.0f;

if (global_id >= element_count) {
    return;
}

output[global_id] = input[global_id] * 2.0f;
```

## Host-side launch

Lesson 14's runtime turns:

```cpp
LaunchConfig::cover(10, 4)
```

into the Metal call:

```objective-c++
[encoder
    dispatchThreadgroups:MTLSizeMake(3, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
```

The two sizes mean different things:

```text
dispatchThreadgroups     number of groups
threadsPerThreadgroup    number of threads inside each group
```

The GPU receives:

```text
3 groups × 4 threads = 12 threads
```

## Binding contract

The MSL kernel declares:

```text
buffer 0 → input
buffer 1 → output
buffer 2 → global-ID records
buffer 3 → group-ID records
buffer 4 → local-ID records
buffer 5 → active/padded flags
buffer 6 → logical element count
```

The host must use matching indexes:

```cpp
dispatch
    .bind_buffer(0, input)
    .bind_buffer(1, output)
    .bind_buffer(2, global_ids)
    .bind_buffer(3, group_ids)
    .bind_buffer(4, local_ids)
    .bind_buffer(5, active_flags)
    .bind_u32(6, element_count);
```

The compiler/runtime contract must match exactly.

## Why no barrier or atomic operation is needed

Every thread writes different locations:

```text
thread global 0  → geometry[0], output[0]
thread global 1  → geometry[1], output[1]
thread global 2  → geometry[2], output[2]
...
```

No two threads update the same element. They do not need to communicate.

Therefore this kernel needs:

```text
no threadgroup shared memory
no barrier
no atomic operation
```

Geometry alone assigns independent work.

## Ordered output does not imply ordered execution

The printed table appears in global-ID order:

```text
0, 1, 2, 3, ...
```

This does not mean thread zero executed first.

The GPU may execute threads and groups in a different or overlapping order.
The table is ordered because:

```text
thread 7 writes its information into geometry[7]
thread 2 writes its information into geometry[2]
thread 9 writes its information into geometry[9]
```

The CPU prints the completed array from index zero upward after waiting for
the event.

Never use an assumed thread execution order for correctness.

## Complete architecture

```text
Host program
    │
    ├── logical elements = 10
    ├── threads per group = 4
    └── LaunchConfig::cover() → 3 groups
             │
             ▼
Lesson 14 MiniGPU Queue
             │
             ▼
Metal dispatches 3 × 4 real GPU threads
             │
             ├── Metal provides global/group/local IDs
             ├── all 12 threads record their geometry
             └── only global IDs 0–9 process tensor data
             │
             ▼
Event::wait()
             │
             ▼
CPU reads and verifies the results
```

## Expected output

```text
Metal device: Apple ...
Logical elements: 10
Launch: 3 groups x 4 threads = 12 launched threads
Padded threads: 2

global    group     local     status      output
0         0         0         active      2
1         0         1         active      4
2         0         2         active      6
3         0         3         active      8
4         1         0         active      10
5         1         1         active      12
6         1         2         active      14
7         1         3         active      16
8         2         0         active      18
9         2         1         active      20
10        2         2         padded      -
11        2         3         padded      -

PASS: real Metal thread IDs and padded threads matched the expected geometry.
```

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
    -o lesson15

./lesson15
```

Lesson 15 deliberately reuses Lesson 14's runtime instead of copying it.
This lesson adds a kernel and an experiment, not another accelerator
runtime.

## Experiments

Change:

```cpp
element_count
threads_per_group
```

Predict the geometry before running:

```text
5 elements, 4 threads/group  → 2 groups, 8 launched, 3 padded
8 elements, 4 threads/group  → 2 groups, 8 launched, 0 padded
9 elements, 4 threads/group  → 3 groups, 12 launched, 3 padded
10 elements, 8 threads/group → 2 groups, 16 launched, 6 padded
```

For each case, verify:

```text
group_id  = global_id / threads_per_group
local_id  = global_id % threads_per_group
```

## Main takeaways

1. A launch is divided into fixed-size threadgroups.
2. Every thread receives global, group, and local IDs from Metal.
3. Local IDs restart from zero inside every group.
4. In one dimension, global ID equals group ID times group size plus local ID.
5. Ceiling division launches enough complete groups to cover all elements.
6. Complete groups can create extra padded threads.
7. Padded threads must return before accessing logical tensor buffers.
8. Ordered output does not prove ordered GPU execution.
9. Independent per-element work needs no barrier or atomic operation.
10. Launch geometry prepares us for real threadgroup memory, barriers, and
    reductions.

## What to understand versus skim

Understand:

```text
logical elements
groups
threads per group
global ID
group ID
local ID
padded threads
bounds check
```

Skim:

```text
output table formatting
floating-point conversion used for diagnostic IDs
repetitive buffer initialization
exception-printing boilerplate
```

## Next lesson

Lesson 16 will examine real GPU memory:

```text
Apple unified physical memory
Metal buffer resources
CPU and GPU visibility
storage modes
private versus shared storage
copying
events and synchronization
```
