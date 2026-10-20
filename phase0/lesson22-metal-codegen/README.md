# Lesson 22 — Typed IR to generated Metal

## Outcome

Lesson 21 created a typed compiler program:

```text
%0 = input @a : tensor<4xf32>
%1 = input @b : tensor<4xf32>
%2 = constant_f32 2.0 : f32
%3 = multiply %1, %2 : tensor<4xf32>
%4 = add %0, %3 : tensor<4xf32>
%5 = relu %4 : tensor<4xf32>
return %5
```

Lesson 22 converts it into:

```text
generated MSL source
buffer-binding metadata
output metadata
launch metadata
```

Then it uses the existing MiniGPU runtime and Apple's Metal compiler to run
that generated program on the real Apple GPU.

```text
typed MiniIR Program
        ↓ lower_to_metal()
MetalKernelPlan
    ├── MSL source
    ├── input buffer ABI
    ├── output buffer ABI
    ├── element-count ABI
    └── launch metadata
        ↓ compile_metal()
Apple Metal compiler
        ↓
MetalExecutable
    ├── immutable copy of the plan
    └── exact compiled MiniGPU Kernel
        ↓ executable.make_dispatch()
        ↓ executable.make_launch()
MiniGPU Queue
        ↓
Apple GPU
```

The main result remains:

```text
[0, 5, 0, 18]
```

## Reusing the earlier layers

Lesson 22 adds only:

```text
metal_backend.hpp
metal_backend.cpp
demo.mm
README.md
```

It reuses:

```text
Lesson 21
    typed IR, builder and verifier

Lesson 14
    MiniGPU Device, Buffer, Kernel,
    Dispatch, Queue and Event
```

The build compiles those earlier implementations together with this lesson.
They are not copied into the new directory.

## What lowering means here

Lowering translates one representation into a more target-specific
representation.

Input:

```text
%3 = multiply %1, %2 : tensor<4xf32>
```

Generated MSL:

```metal
const float v3 = v1 * v2;
```

The SSA dependency remains the same:

```text
%1 and %2 produce %3
```

Only its representation changes:

```text
MiniIR Value IDs
        ↓
MSL local variables
```

## One Program is one elementwise kernel region

Lesson 22 defines one straight-line elementwise `Program` as one GPU kernel
region.

Therefore:

```text
%2 = constant
%3 = multiply
%4 = add
%5 = ReLU
```

becomes local calculations inside one MSL function.

This lesson is not running an optimization pass that discovers several
independent kernels and decides to fuse them. It is directly translating
the one elementwise region it receives.

Later optimization work will decide:

```text
which graph operations belong in one region
which regions should remain separate
which constants can fold
which unused operations can disappear
```

## The generated MSL

For the running example, the compiler emits:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void miniir_elementwise(
    device const float* input_0 [[buffer(0)]],
    device const float* input_1 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    const float v0 = input_0[global_id];
    const float v1 = input_1[global_id];
    const float v2 = 2.0f;
    const float v3 = v1 * v2;
    const float v4 = v0 + v3;
    const float v5 = max(v4, 0.0f);

    output[global_id] = v5;
}
```

This string was not handwritten in `demo.mm`.

`lower_to_metal()` generated it by walking the operations in the typed
program.

## How each IR operation lowers

| MiniIR operation | Generated MSL |
|---|---|
| Tensor input | `input_N[global_id]` |
| Scalar input | `input_N[0]` |
| `constant_f32` | floating-point literal |
| `multiply` | `left * right` |
| `add` | `left + right` |
| `relu` | `max(input, 0.0f)` |
| `return` | `output[global_id] = value` |

Inputs become loads.

Arithmetic operations become local scalar expressions.

Return becomes the final output-buffer store.

## SSA values become lane-local variables

The typed IR has:

```text
%3 = multiply %1, %2
```

The MSL has:

```metal
const float v3 = v1 * v2;
```

Every active GPU lane has its own runtime instance of:

```text
v0
v1
v2
v3
v4
v5
```

For lane zero, `v0` might be `-3`.

For lane one, its separate `v0` might be `1`.

The variables are conceptually private to each lane. The Metal compiler will
usually keep short-lived values like these in registers, although the final
hardware placement remains Apple's decision.

## Why there are no intermediate tensor buffers

The generated kernel does not allocate:

```text
temporary tensor for b × 2
temporary tensor for a + scaled
```

Inside one lane:

```text
load a element
load b element
calculate multiply
calculate add
calculate ReLU
store one output element
```

The intermediate values are scalar MSL locals, not full tensors.

This follows directly from the current contract that one MiniIR `Program`
describes one elementwise kernel region.

## The binding ABI

The generated source and the CPU runtime must agree about every binding
index.

For the example:

```text
buffer(0) → input @a
buffer(1) → input @b
buffer(2) → output
buffer(3) → uint element_count
```

The generated plan prints:

```text
function: miniir_elementwise
buffer(0) input @a -> %0 : tensor<4xf32> [4 element(s)]
buffer(1) input @b -> %1 : tensor<4xf32> [4 element(s)]
buffer(2) output <- %5 : tensor<4xf32>
buffer(3) element_count = 4
launch: 4 work item(s), 64 thread(s) per group
```

ABI means the agreement between two layers.

Here it is the agreement between:

```text
generated MSL parameters
        ↕
MiniGPU Dispatch bindings
```

If the compiler says input `a` is `buffer(0)`, the runtime must bind `a` at
index zero.

## Deterministic input bindings

Input operations receive slots in IR order:

```text
first Input  → buffer(0)
second Input → buffer(1)
third Input  → buffer(2)
```

Then:

```text
output        → next buffer
element_count → following buffer
```

The same verified program therefore always receives the same source and ABI.

That determinism will later matter for compiled-kernel caching.

## Why user input names do not become MSL names

The IR may contain:

```text
input @a
input @scale
```

The generated MSL uses:

```text
input_0
input_1
```

User names remain in host-side metadata:

```text
@a → buffer(0)
@scale → buffer(1)
```

Synthetic MSL identifiers avoid:

```text
reserved Metal words
identifier escaping
name collisions
target-language syntax problems
```

## Scalar broadcasting becomes indexing

Consider:

```text
x     : tensor<2x3xf32>
scale : f32
bias  : f32
```

For:

```text
relu(x × scale + bias)
```

the compiler generates:

```metal
const float v0 = input_0[global_id];
const float v1 = input_1[0];
const float v2 = input_2[0];
```

Tensor input:

```text
x → different element for each global_id
```

Scalar inputs:

```text
scale → element zero for every lane
bias  → element zero for every lane
```

No six-element copy of either scalar is created.

The real GPU test produces:

```text
x     = [-2, -1, 0, 1, 2, 3]
scale = 2
bias  = -1

relu(x × 2 - 1)
    = [0, 0, 0, 1, 3, 5]
```

## Flattened tensor indexing

The MSL kernel uses one index:

```metal
global_id
```

Therefore:

```text
tensor<4xf32>    → 4 flat elements
tensor<2x3xf32>  → 6 flat elements
```

Lesson 22 assumes a simple contiguous flat buffer.

It does not yet generate multidimensional address calculations or understand
strides and views.

For this elementwise kernel:

```text
logical tensor index
        ↓ flatten
global_id
        ↓
buffer[global_id]
```

## One iteration domain

The output type determines the number of logical work items:

```text
f32                 → 1
tensor<4xf32>       → 4
tensor<2x3xf32>     → 6
```

Every non-scalar value must use that same tensor shape.

Why?

The generated kernel evaluates every recorded operation for every output
index.

If the program contained an unused `tensor<3xf32>` while producing a
`tensor<4xf32>`, lane three could try to read beyond the three-element
buffer.

Dead-code elimination is not implemented yet, so Lesson 22 rejects mixed
non-scalar iteration domains.

## The padded-thread guard

The example has four logical elements but launches:

```text
1 group × 64 threads = 64 threads
```

Only lanes zero through three are active.

The generated guard protects the buffers:

```metal
if (global_id >= element_count) {
    return;
}
```

The remaining sixty padded lanes return before reading or writing tensor
memory.

## Why `element_count` uses `uint`

Lesson 14 MiniGPU currently supports:

```cpp
bind_u32(index, value);
```

The generated MSL therefore declares:

```metal
constant uint& element_count
```

Before code generation, Lesson 22 checks that the output element count fits
inside `std::uint32_t`.

This is an example of a backend limitation influencing compiler validation.

## `MetalKernelPlan`

`lower_to_metal()` does not directly launch the GPU.

It returns a plan containing:

```text
function name
generated source
input bindings
output binding
element-count binding
output Value ID
output Type
logical work count
threads per group
```

This preserves the compiler/runtime boundary:

```text
Compiler backend
    decides what source and metadata are required

Runtime
    supplies actual buffers and submits the work
```

The plan exposes read-only accessors. Code outside the backend cannot change
one binding or shape while leaving the generated source unchanged.

That matters because source and metadata describe one ABI together. For
example, this would be unsafe:

```text
generated source reads input[global_id]
        but
mutated metadata accepts only a one-element input buffer
```

An immutable plan prevents that disagreement.

## `MetalExecutable` permanently couples plan and kernel

Compilation uses:

```cpp
MetalExecutable executable =
    compile_metal(device, plan);
```

`compile_metal()` invokes Apple's compiler using the plan's exact function
name and source. It then returns one object that owns:

```text
the immutable MetalKernelPlan
the MiniGPU Kernel compiled from that plan's source
```

The raw kernel is not exposed by this API. Dispatches are created through
the executable:

```cpp
Dispatch dispatch =
    executable.make_dispatch(inputs, output);
```

This is stronger than merely checking that two kernels have the same
function name. Different generated programs currently use the same function
name, so name equality alone would not prove that their binding ABIs match.

The executable makes the safe relationship structural:

```text
one generated plan
        +
kernel compiled from that exact plan
        ↓
one reusable MetalExecutable
```

## Apple's compiler still handles machine code

Our compiler generates:

```text
MSL source
```

Then:

```cpp
MetalExecutable executable =
    compile_metal(device, plan);
```

invokes Apple's compiler:

```text
generated MSL
    ↓ Apple compiler
Apple GPU machine code
    ↓ pipeline creation
MiniGPU Kernel
```

We do not generate Apple GPU machine instructions ourselves.

## Constructing the dispatch

`MetalExecutable::make_dispatch()` connects the executable's metadata to real
buffers and uses its already-compiled kernel.

Conceptually:

```cpp
Dispatch dispatch(kernel);

dispatch
    .bind_buffer(0, a)
    .bind_buffer(1, b)
    .bind_buffer(2, output)
    .bind_u32(3, element_count);
```

It validates:

```text
every required input exists
no unknown input name was provided
each scalar input buffer has one element
each tensor input buffer has the declared number of elements
the output buffer has the required size
```

The exact kernel/plan pairing does not need a runtime name check because it
was established by `compile_metal()` and cannot be changed afterward.

The compiler describes the ABI.

The runtime enforces it before submitting unsafe GPU work.

## Launch construction

The plan contains:

```text
logical work items = 4
threads per group  = 64
```

`MetalExecutable::make_launch()` produces:

```cpp
LaunchConfig::cover(4, 64);
```

which becomes:

```text
1 group × 64 threads
```

The fixed choice of 64 is a simple baseline, not an optimized schedule.

Later performance work can select threadgroup sizes based on the operation
and hardware.

## Execution and synchronization

The demo uses shared MiniGPU buffers:

```text
CPU writes input buffers
        ↓
compile_metal() creates the coupled executable
        ↓
executable constructs Dispatch from its metadata
        ↓
Queue::submit()
        ↓
Apple GPU executes
        ↓
Event::wait()
        ↓
CPU reads output buffer
```

The CPU must not read the output before `wait()` completes.

## Error layers

Different layers report different failures.

### `VerificationError`

The typed MiniIR program itself is malformed:

```text
undefined operand
wrong stored result type
missing Return
```

### `CodegenError`

The program is valid MiniIR but unsupported by this Metal backend:

```text
f16 values
more than uint32-sized output
different non-scalar iteration domains
```

### `std::invalid_argument`

The runtime buffers do not satisfy the generated plan:

```text
missing input
unknown input
wrong input size
wrong output size
```

### `MetalError`

Apple's Metal compiler or GPU execution reports a failure:

```text
invalid generated MSL
pipeline creation failure
GPU command failure
```

## What the demo checks

The executable verifies:

```text
exact deterministic generated MSL
exact deterministic ABI and launch metadata
compile-time read-only plan access
restricted plan and executable construction
the executable retains the exact generated plan
dispatch and execution through the coupled MetalExecutable
input binding order
output and element-count slots
SSA-to-MSL expression lowering
tensor inputs use [global_id]
scalar inputs use [0]
padded-thread bounds guard
f16 backend rejection
mixed tensor-domain rejection
oversized u32 domain rejection
missing runtime input rejection
unknown runtime input rejection
wrong input size rejection
wrong output size rejection
Apple Metal source compilation
real GPU execution
GPU result matches the CPU reference
runtime scalar broadcasting
```

The two real GPU results are:

```text
relu(a + b × 2)
    → [0, 5, 0, 18]

relu(x × scale + bias)
    → [0, 0, 0, 1, 3, 5]
```

## Build and run

From the workspace root:

```bash
xcrun clang++ \
  -std=c++20 \
  -O2 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fobjc-arc \
  outputs/infer-lab/phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  outputs/infer-lab/phase0/lesson22-metal-codegen/metal_backend.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson22-metal-codegen/demo.mm \
  -framework Foundation \
  -framework Metal \
  -o work/lesson22_metal_codegen

./work/lesson22_metal_codegen
```

Metal GPU access may require running outside a restricted sandbox.

## Expected key output

```text
Metal device: Apple M4 Pro
Launch: 1 group(s) x 64 threads = 64 launched for 4 logical elements
Main GPU output: [0, 5, 0, 18]
Scalar-broadcast GPU output: [0, 0, 0, 1, 3, 5]

Codegen, metadata, binding, launch and GPU checks: PASS
PASS: generated MSL ran on the real Apple GPU.
```

## Main takeaways

1. Lowering translates typed MiniIR into target-specific MSL.
2. SSA values become lane-local MSL variables.
3. Typed inputs determine whether generated loads use `[0]` or
   `[global_id]`.
4. Scalar broadcasting becomes an indexing decision, not a tensor copy.
5. The output type determines the flattened iteration count.
6. Generated source and host bindings share an explicit ABI.
7. Launch metadata and the bounds guard make padded lanes safe.
8. An immutable plan keeps generated source and ABI metadata consistent.
9. `MetalExecutable` keeps the exact compiled kernel coupled to that plan.
10. MiniGPU executes the generated plan but does not understand MiniIR.
11. Apple's compiler still converts MSL into real machine code.
12. This is the first end-to-end compiler path from IR to real GPU output.

## What to understand versus skim

Understand:

```text
typed Program → MetalKernelPlan
SSA Value → MSL local variable
Input operation → buffer binding
scalar input → input[0]
tensor input → input[global_id]
Return → output store
output type → work-item count
metadata ABI → Dispatch bindings
immutable plan → source and metadata cannot drift
compile_metal() → exact plan + exact Kernel
MSL source → Apple compiler → Kernel
submit → wait → safe CPU read
```

Skim:

```text
string indentation
float-literal formatting
host vector printing
exception helper templates
negative-test construction
Objective-C autorelease pool
```

## What remains deferred

Lesson 22 does not implement:

```text
f16 GPU execution
casts or dtype promotion
general broadcasting
dynamic shapes
strides, views or transposes
multidimensional address calculations
matmul, reductions or softmax
optimization passes
constant folding
dead-code elimination
fusion decisions across kernel regions
kernel caching
autotuning
MiniTensor operator syntax
```

Lesson 23 will introduce explicit compiler optimization passes and make the
before-versus-after IR transformations visible.
