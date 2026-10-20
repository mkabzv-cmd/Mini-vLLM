# Lesson 28 — Contiguous materialization and buffer reuse

## The two ideas

```text
materialization:
view layout → new compact row-major storage

reuse:
write a later result into storage we already allocated
```

Lesson 27 produced a contiguous ReLU output. This lesson isolates the copying
part so any supported tensor view can become an ordinary compact tensor.

## What we built

```cpp
ContiguousMaterializer materializer(device);

RuntimeTensor copy =
    materializer.run(queue, input);
```

`run()`:

```text
allocates independent output
submits the Metal copy kernel
waits for completion
returns safe CPU-readable output
```

For explicit reuse:

```cpp
Event finished =
    materializer.enqueue_into(
        queue,
        input,
        reusable_output
    );

finished.wait();
```

`enqueue_into()` does not allocate output and does not wait.

## What “materialize” means

Suppose a transpose view is:

```text
shape   = [4,3]
strides = [1,4]
offset  = 0

logical values:
[0,4,8, 1,5,9, 2,6,10, 3,7,11]

physical source storage:
[0,1,2,3,4,5,6,7,8,9,10,11]
```

The materialized result is:

```text
shape   = [4,3]
strides = [3,1]
offset  = 0

physical output storage:
[0,4,8, 1,5,9, 2,6,10, 3,7,11]
```

The logical values did not change. Their physical arrangement changed.

```text
non-contiguous interpretation
            ↓ GPU copy
compact physical representation
```

## The Metal copy kernel

Each lane owns one logical output index:

```text
global_id = logical index
```

It converts that logical index into the input's physical index using shape,
strides, and offset:

```metal
uint remaining = global_id;
uint input_index = offset;

for (uint axis = rank; axis > 0; --axis) {
    const uint current = axis - 1;
    const uint coordinate =
        remaining % shapes[current];

    remaining /= shapes[current];

    input_index +=
        coordinate * strides[current];
}
```

Then the copy itself is:

```metal
output[global_id] = input[input_index];
```

The input address may jump around. The output address is always dense.

## Why `run()` always copies

Even if the input already has canonical strides and offset zero, this
lesson's `run()` returns independent storage.

```cpp
RuntimeTensor copy =
    materializer.run(queue, already_contiguous);
```

Afterward:

```text
input Buffer ───── original allocation
output Buffer ──── different allocation
```

Changing one does not change the other.

This explicit rule is easier to reason about:

```text
run() means materialize an independent copy
```

Production frameworks often add a separate optimization that returns the
original handle when no copy is needed. We defer that optimization until
aliasing and liveness decisions are made deliberately.

## Contiguous does not necessarily mean offset zero

Lesson 25 defines contiguity from strides.

A complete-row slice may have:

```text
shape   = [2,4]
strides = [4,1]
offset  = 4
contiguous = yes
```

Its values are adjacent, but they begin at `storage[4]`.

Lesson 28's output contract is stricter:

```text
canonical row-major strides
offset = 0
independent storage
```

Therefore `run()` still copies that row slice.

## Reusing an output allocation

Repeated allocation is expensive in a real inference loop.

Instead of:

```text
allocate → copy → discard
allocate → copy → discard
allocate → copy → discard
```

we can do:

```text
allocate workspace once
copy result A into workspace
wait until safe
copy result B into the same workspace
```

The API is:

```cpp
Event enqueue_into(
    Queue& queue,
    const RuntimeTensor& input,
    RuntimeTensor& output
);
```

The output must have:

```text
exactly the same type and shape as the input
canonical row-major strides
offset zero
enough storage capacity
different storage from the input
```

## Logical size versus capacity

The demo creates:

```text
logical output elements = 6
buffer capacity         = 8
```

`RuntimeTensor::numel()` answers how many values belong to the tensor.

`RuntimeTensor::storage_size()` answers how many `float` slots the backing
buffer can hold.

The extra capacity is allowed:

```text
storage: [value value value value value value canary canary]
          └──────── six logical values ────────┘
```

This is the beginning of a memory-pool idea: a larger allocation can satisfy
a smaller request.

## Why the canaries matter

Six values with four threads per group launch:

```text
ceil(6 / 4) × 4 = 8 lanes
```

The output's slots `6` and `7` are initialized to `-777`.

The kernel starts with:

```metal
if (global_id >= element_count) {
    return;
}
```

After copying, the canaries must still be `-777`.

Unlike an out-of-bounds-only test, this deterministically proves that padded
lanes did not write.

## Why input and output may not share storage

Consider materializing a transpose into its own original buffer.

One lane may write:

```text
output[1]
```

before another lane has read the old value from:

```text
input[1]
```

That creates a read/write race and can corrupt the permutation.

RuntimeTensor objects can have different addresses while still carrying
copies of the same shared `Buffer` handle. Therefore this is not enough:

```cpp
&input != &output
```

Lesson 28 adds:

```cpp
buffer.shares_storage_with(other_buffer);
```

It compares the underlying Buffer allocation identity.

`enqueue_into()` conservatively rejects any shared allocation, even if two
regions might happen not to overlap.

## Asynchronous reuse

`enqueue_into()` returns immediately:

```cpp
Event finished =
    materializer.enqueue_into(...);
```

At that moment the GPU may still be:

```text
reading the input
writing the reusable output
```

Do not let the CPU read or modify either buffer until:

```cpp
finished.wait();
```

Do not reuse the destination on another queue without waiting either. Our
MiniGPU does not yet expose cross-queue GPU dependencies.

## Same-queue chaining

Commands submitted to one queue execute in FIFO order.

Therefore:

```cpp
Event first =
    materializer.enqueue_into(
        queue,
        view,
        temporary
    );

Event second =
    materializer.enqueue_into(
        queue,
        temporary,
        output
    );

second.wait();
```

The CPU did not wait between the two submissions.

The second GPU command still reads `temporary` only after the first command
has written it because both commands use the same queue.

Waiting for `second` also means the earlier first command has completed.

## Buffer lifetime

`enqueue_into()` binds copied Buffer handles.

MiniGPU moves those handles into the returned `Event`, so keep that Event
until calling `wait()`. While the Event exists, the Metal allocations stay
alive even if the original RuntimeTensor handles are destroyed.

This remains safe:

```text
create temporary input view
submit copy
destroy temporary RuntimeTensor handles
wait for Event
read persistent output
```

This solves lifetime. It does not make early CPU access safe; synchronization
still requires the Event.

## What buffer reuse means here

This lesson implements manual, explicit reuse:

```text
the caller owns one destination
the caller knows when it is safe to overwrite
```

It does not yet implement an automatic allocator or free list.

A production pool additionally tracks:

```text
capacity
dtype and alignment
whether commands are still using the allocation
which tensors are still alive
which queue or event makes reuse safe
```

Hidden reuse would be dangerous. If a runtime silently reused storage while
an older returned tensor was still alive, that older tensor's values would
unexpectedly change.

## Validation

Before dispatch, the runtime checks:

```text
input rank is at most four
input layout values fit uint32 GPU indexing
padded launch size fits uint32
output type exactly matches input type
output uses canonical strides
output offset is zero
input and output allocations are distinct
threads per group is positive
```

## What the demo covers

```text
contiguous input still receives independent storage
contiguous input with a nonzero offset
transpose
composed transpose plus stepped slice
rank-zero scalar with a nonzero offset
rank-four transpose
oversized reusable output with canaries
two copies reusing one allocation
same-queue chaining without a CPU wait between commands
pending command retaining temporary input storage
alias rejection
destination validation
rank and uint32 launch validation
```

## What to read

Read:

1. The MSL `output[global_id] = input[input_index]`.
2. `validate_output()`.
3. `Buffer::shares_storage_with()`.
4. `enqueue_into()`.
5. The workspace/canary section of `demo.cpp`.

Skim:

```text
printing helpers
repeated expected vectors
binding constants
exception-test helper
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
  outputs/infer-lab/phase0/lesson28-contiguous-buffer-reuse/contiguous.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson28-contiguous-buffer-reuse/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson28_contiguous_buffer_reuse

./work/lesson28_contiguous_buffer_reuse
```

## Main takeaways

1. Materialization changes physical layout, not logical values.
2. `run()` always returns independent compact storage.
3. `enqueue_into()` lets the caller reuse an existing allocation.
4. Logical element count and buffer capacity are different concepts.
5. Padded lanes must not touch canary capacity.
6. Input and output aliasing is conservatively rejected.
7. An Event retains storage lifetime but must still be waited on before CPU
   access or unsafe reuse.
8. Same-queue FIFO can order dependent GPU commands without a CPU wait
   between every command.

## Deliberately deferred

```text
automatic buffer pools
reuse across arbitrary shapes and dtypes
liveness analysis
cross-queue fences
compiler-inserted materialization
blit-copy fast paths
f16
rank greater than four
64-bit indexing
negative and zero strides
```

## Next lesson

Lesson 29 will begin the most important transformer operation:

```text
matrix multiplication
```

We will first build a correct naive Metal matmul using `M`, `N`, and `K`,
then use it to understand why tiling is needed.
