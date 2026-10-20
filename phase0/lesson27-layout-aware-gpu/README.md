# Lesson 27 — A GPU kernel that understands tensor layouts

## The one idea

Before this lesson, CPU code understood:

```text
shape + strides + offset
```

Now a real Metal kernel uses that metadata too.

```text
GPU lane's logical index
        ↓
logical coordinates
        ↓
shape + strides + offset
        ↓
physical input-buffer index
```

This is what lets one kernel read contiguous tensors, transposes, slices, and
stepped views correctly.

## What we built

```cpp
LayoutAwareRelu relu(device);

RuntimeTensor output =
    relu.run(queue, input_view);
```

The operation computes:

```text
output = ReLU(input_view)
```

where:

```text
ReLU(x) = max(x, 0)
```

The input may have any Lesson 26 layout with rank `0` through `4`. The output
is a new contiguous tensor.

## Why the old indexing is insufficient

Our earlier elementwise kernels used:

```metal
float value = input[global_id];
```

That assumes logical value `i` is stored at physical index `i`.

It works for:

```text
shape=[3,4]
strides=[4,1]
offset=0
```

It fails for this column slice:

```text
shape=[3,2]
strides=[4,1]
offset=1
```

The slice's six logical values use physical indexes:

```text
1, 2, 5, 6, 9, 10
```

A naive kernel would incorrectly read:

```text
0, 1, 2, 3, 4, 5
```

## One concrete lane

For the column slice:

```text
shape   = [3,2]
strides = [4,1]
offset  = 1
```

suppose GPU lane `global_id = 4`.

The lane first treats `4` as a logical row-major index.

Convert it to coordinates from right to left:

```text
remaining = 4

column = 4 % 2 = 0
remaining = 4 / 2 = 2

row = 2 % 3 = 2
remaining = 2 / 3 = 0
```

So:

```text
logical index 4 → coordinate [2,0]
```

Now apply the layout:

```text
physical index
    = offset + row × row_stride + column × column_stride
    = 1 + 2 × 4 + 0 × 1
    = 9
```

The lane therefore computes:

```metal
output[4] = max(input[9], 0.0f);
```

## The central Metal code

```metal
uint remaining = global_id;
uint storage_index = offset;

for (uint axis = rank; axis > 0; --axis) {
    const uint current = axis - 1;

    const uint coordinate =
        remaining % shapes[current];

    remaining /= shapes[current];

    storage_index +=
        coordinate * strides[current];
}

output[global_id] =
    max(input[storage_index], 0.0f);
```

This is the GPU version of Lesson 25's
`TensorLayout::storage_index_from_linear()`.

## Why conversion starts from the right

In row-major order, the rightmost coordinate changes fastest.

For shape `[3,2]`, flat logical indexes mean:

| Logical index | Coordinate |
|---:|---|
| 0 | `[0,0]` |
| 1 | `[0,1]` |
| 2 | `[1,0]` |
| 3 | `[1,1]` |
| 4 | `[2,0]` |
| 5 | `[2,1]` |

Modulo finds the rightmost coordinate:

```text
coordinate = remaining % dimension
```

Integer division removes that coordinate:

```text
remaining /= dimension
```

Then the loop continues toward the left.

## How layout metadata reaches Metal

MiniGPU already supports:

```cpp
dispatch.bind_u32(index, value);
```

Lesson 27 uses individual constant bindings for:

```text
element count
rank
offset
4 shape entries
4 stride entries
```

The CPU packs unused axes as:

```text
shape  = 1
stride = 0
```

It also sends stride `0` for a used axis whose shape is `1`, because that
coordinate can never advance and the stride cannot affect an address.

Then MiniGPU sends the values through Metal's constant-buffer arguments.

The GPU does not discover this metadata automatically. The CPU must bind the
correct values for every dispatch. Strides and offset count `float` elements,
not bytes, which matches Metal pointer indexing.

`bind_u32()` copies four bytes into the Metal command encoder using
`setBytes`. It does not allocate another user-visible `minigpu::Buffer`.

Production runtimes often place metadata in compact parameter buffers or
generate rank-specialized kernels. Individual bindings keep our first
implementation visible and easy to inspect.

## Why rank four

The teaching kernel supports up to four dimensions.

That already represents the important attention layout:

```text
[batch, heads, tokens, head_dimension]
```

The loop itself is generic. Four is the fixed amount of metadata exposed in
this first MSL function signature; `layout_kernel_max_rank` describes that ABI
and is not a freely configurable runtime setting.

Modulo and division make the mapping easy to understand, but they are not
always the fastest implementation. Later compiler passes can specialize
common ranks and layouts after correctness is established.

## Why the output is new and contiguous

The kernel does not modify the input view.

It allocates:

```text
same logical type
canonical row-major strides
offset 0
separate Metal buffer
```

Then every lane writes:

```metal
output[global_id]
```

This has two advantages:

1. The logical ordering of a transpose becomes observable in the output.
2. Reading and writing overlapping views cannot corrupt the calculation.

If a transpose were processed in place, one lane could overwrite a value
before another lane had read it.

No barrier or atomic is needed here. Every lane only reads the shared input
and owns one distinct `output[global_id]`.

Lesson 28 will use the same indexing idea to implement an explicit
`contiguous()` materialization operation.

## Compile once, run many layouts

The constructor compiles the MSL once:

```cpp
LayoutAwareRelu relu(device);
```

The same Metal pipeline then runs on:

```text
contiguous matrix
column slice
transpose
stepped slice
composed view
rank-three tensor
scalar
```

Only the bound metadata changes.

This is an early example of separating:

```text
compiled kernel code
from
runtime tensor metadata
```

## Padded GPU lanes

Suppose the input view has six logical values and we choose four threads per
group:

```text
groups = ceil(6 / 4) = 2
launched lanes = 2 × 4 = 8
```

Lanes `0` through `5` own real output values.

Lanes `6` and `7` are padding, so the kernel begins with:

```metal
if (global_id >= element_count) {
    return;
}
```

This guard happens before any shape lookup, input read, or output write.

The public demo exercises a padded launch and the guard is deliberately the
first code in the MSL kernel. A future lower-level kernel test can use extra
canary storage to detect deliberate removal of the guard deterministically.

## Synchronization

`run()` submits the command and receives an `Event`:

```cpp
Event finished = queue.submit(...);
```

The output buffer is still being written asynchronously at that point.

This lesson's simple synchronous API calls:

```cpp
finished.wait();
```

before returning the `RuntimeTensor`.

Therefore the caller may immediately use:

```cpp
output.to_host();
```

Later, the inference engine will keep multiple operations asynchronous and
wait only where dependencies require it.

## The uint32 boundary

This Metal kernel uses `uint`, which is 32-bit.

Before dispatch, CPU code verifies that:

```text
logical element count
rank
offset
dimensions
used strides
highest physical input index
total padded launch size
```

fit the kernel's indexing model.

This is an important runtime responsibility:

```text
host metadata and device arithmetic must agree
```

## Scalar behavior

A scalar has:

```text
shape=[]
rank=0
one logical value
```

The coordinate loop executes zero times, so:

```text
storage_index = offset
```

One GPU lane reads the scalar and writes one output value.

## What the demo covers

The demo checks:

```text
contiguous input
rank-one stepped and offset input
non-contiguous column slice
transpose logical order
stepped slice
composed view with an offset
rank-three transpose
rank-four transpose
rank-zero scalar with a nonzero offset
padded-launch execution
input remains unchanged
output is contiguous
rank and launch validation
```

The transpose test is especially important. A naive `input[global_id]` kernel
would return the original physical order instead of the transpose's logical
order.

## What to read

Read:

1. The MSL loop in `kernel_source`.
2. `pack_layout()` on the CPU.
3. The `bind_u32()` loop in `LayoutAwareRelu::run()`.
4. `finished.wait()`.
5. The transpose and column-slice tests.

Skim:

```text
printing helpers
repetitive expected vectors
exception-test helper
binding-index constants
```

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
  outputs/infer-lab/phase0/lesson26-tensor-views/tensor_views.cpp \
  outputs/infer-lab/phase0/lesson27-layout-aware-gpu/layout_relu.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson27-layout-aware-gpu/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson27_layout_aware_gpu

./work/lesson27_layout_aware_gpu
```

## Main takeaways

1. A lane starts with a flat logical index.
2. Shape converts that index into coordinates.
3. Strides and offset convert coordinates into a physical buffer index.
4. The same compiled kernel can consume many layouts.
5. A guard protects padded lanes.
6. A fresh contiguous output avoids input/output aliasing.
7. The CPU validates and binds metadata before dispatch.
8. Waiting on the Event makes the returned output safe to read.

## Deliberately deferred

Later lessons will add:

```text
explicit contiguous() copies
buffer reuse
asynchronous operator chaining
compiler-generated layout-aware kernels
more than four dimensions
64-bit indexing where required
negative and zero strides
```

## Next lesson

Lesson 28 will implement:

```text
RuntimeTensor contiguous(view)
```

It will copy any supported layout into canonical row-major storage and begin
reusing temporary buffers instead of allocating a new one every time.
