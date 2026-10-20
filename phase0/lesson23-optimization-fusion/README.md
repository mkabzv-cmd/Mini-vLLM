# Lesson 23 — Optimization passes and linear elementwise fusion

## Outcome

Lesson 22 could lower one finished `Program` into one Metal kernel.

Lesson 23 first optimizes the scale region, then fuses it with two later
regions before Metal lowering:

```text
original scale Program
    ↓ constant folding
folded scale Program
    ↓ dead-code elimination
clean scale Program
        +
shift Program
        +
activation Program
    ↓ linear elementwise fusion
one fused Program
    ↓ Lesson 22 Metal backend
one generated Metal kernel
```

The running calculation is:

```text
result = relu(x × (1 + 1) + 3)
```

For:

```text
x = [-2, -1, 0, 1, 2]
```

the result is:

```text
[0, 1, 3, 5, 7]
```

The demo runs both of these paths on the real Apple GPU:

```text
unfused: 3 dispatches and 2 intermediate buffers
fused:   1 dispatch  and 0 intermediate buffers
```

Both produce the same answer.

This lesson does **not** claim that the fused path is always faster. It only
proves the concrete reduction in dispatch and intermediate-buffer counts for
this example. Real performance claims require measurement.

## Reusing the earlier layers

Lesson 23 adds:

```text
optimizer.hpp
optimizer.cpp
demo.mm
README.md
```

It reuses:

```text
Lesson 21
    typed MiniIR, Program, Operation, verifier and CPU evaluator

Lesson 22
    Metal lowering, generated MSL and MetalExecutable

Lesson 14
    MiniGPU buffers, dispatches, queue and events
```

No new tensor frontend or GPU runtime is introduced here.

## The three separate kernel regions

Before fusion, the calculation is represented by three independent
`Program` objects.

Each `Program` is already a valid Lesson 22 kernel region.

### Stage 1: scale

The first program deliberately contains both foldable and dead work:

```text
%0 = input @x : tensor<5xf32>
%1 = constant_f32 1.0 : f32
%2 = constant_f32 1.0 : f32
%3 = add %1, %2 : f32
%4 = multiply %0, %3 : tensor<5xf32>
%5 = constant_f32 100.0 : f32
%6 = add %0, %5 : tensor<5xf32>
return %4
```

Its useful result is:

```text
scaled = x × (1 + 1)
```

This operation is unused:

```text
%6 = add %0, %5
```

Nothing depends on `%6`, and the program returns `%4`.

### Stage 2: shift

```text
%0 = input @scaled : tensor<5xf32>
%1 = constant_f32 3.0 : f32
%2 = add %0, %1 : tensor<5xf32>
return %2
```

This region calculates:

```text
shifted = scaled + 3
```

### Stage 3: activation

```text
%0 = input @shifted : tensor<5xf32>
%1 = relu %0 : tensor<5xf32>
return %1
```

This region calculates:

```text
result = relu(shifted)
```

## Program versus pipeline

These are different levels of organization.

### `Program`

A `Program` is one straight-line MiniIR kernel region:

```text
inputs
    ↓
operations
    ↓
one Return
```

Lesson 22 accepts one `Program` and generates one Metal kernel from it.

### Pipeline

A pipeline is several `Program` objects connected in order:

```text
scale Program
    ↓ scaled tensor
shift Program
    ↓ shifted tensor
activation Program
    ↓ final result
```

Without fusion, each region is independently lowered and dispatched:

```text
x
    ↓ dispatch 1
[scale kernel]
    ↓ temporary buffer: scaled
    ↓ dispatch 2
[shift kernel]
    ↓ temporary buffer: shifted
    ↓ dispatch 3
[activation kernel]
    ↓
output
```

Fusion changes these kernel-region boundaries.

It does not merely rename an operation.

## Pass versus analysis

A compiler commonly contains both analyses and transformation passes.

### Analysis

An analysis studies a program and produces information about it:

```text
Program
    ↓ inspect
facts about Program
```

It does not have to rewrite the operations.

For example, dead-code elimination first performs a liveness analysis:

```text
Which values contribute to Return?
```

The answer may look like:

```text
%0 live
%1 dead
%2 dead
%3 live
%4 live
%5 dead
%6 dead
```

### Transformation pass

A transformation pass returns another program with the same intended result:

```text
old Program
    ↓ rewrite
new Program
```

Examples in this lesson:

```text
constant folding
dead-code elimination
linear elementwise fusion
```

The original `Program` is not modified.

The demo explicitly checks that optimization leaves the original IR
unchanged.

## The optimizer API

The individual passes are available separately:

```cpp
ConstantFoldingResult fold_constants(
    const Program& program
);

DeadCodeEliminationResult eliminate_dead_code(
    const Program& program
);
```

The normal combined path is:

```cpp
OptimizationResult optimize_elementwise(
    const Program& program
);
```

It performs:

```text
fold_constants()
    ↓
eliminate_dead_code()
```

The result preserves both snapshots:

```cpp
optimized.after_constant_folding
optimized.program
```

It also records explanatory metadata:

```text
which values were folded
which opcode each folded value originally used
old-to-new SSA value mapping
how many values DCE removed
```

`format_optimization_report()` makes that information readable.

## Constant folding

Constant folding calculates work during compilation when all required
operands are already known constants.

The original IR contains:

```text
%1 = constant_f32 1.0 : f32
%2 = constant_f32 1.0 : f32
%3 = add %1, %2 : f32
```

The compiler already knows:

```text
%1 = 1
%2 = 1
```

It therefore knows:

```text
%3 = 1 + 1 = 2
```

After constant folding:

```text
%0 = input @x : tensor<5xf32>
%1 = constant_f32 1.0 : f32
%2 = constant_f32 1.0 : f32
%3 = constant_f32 2.0 : f32
%4 = multiply %0, %3 : tensor<5xf32>
%5 = constant_f32 100.0 : f32
%6 = add %0, %5 : tensor<5xf32>
return %4
```

The runtime no longer needs to execute the addition that produced `%3`.

### Forward constant propagation

MiniIR uses dense SSA IDs:

```text
%0, %1, %2, ...
```

The folding pass walks operations from beginning to end and maintains a
table:

```text
constants[%0] = unknown
constants[%1] = 1
constants[%2] = 1
```

When it reaches:

```text
%3 = add %1, %2
```

both operand entries are known, so it records:

```text
constants[%3] = 2
```

and replaces the operation with:

```text
%3 = constant_f32 2.0
```

This is called forward constant propagation because information about known
constants flows forward through SSA dependencies.

### Nested folding

Suppose the program contains:

```text
%0 = constant_f32 2.0
%1 = constant_f32 3.0
%2 = multiply %0, %1
%3 = constant_f32 -4.0
%4 = relu %3
%5 = add %2, %4
return %5
```

One forward walk can derive:

```text
%2 = 6
%4 = 0
%5 = 6
```

Earlier folded values are immediately available to later operations.

Lesson 23 folds scalar `f32`:

```text
Add
Multiply
ReLU
```

only when all of their operands are known constants.

Tensor calculations involving runtime inputs are not constant-folded.

### Why folding keeps the same SSA IDs

Constant folding replaces an operation but does not delete its result:

```text
before:
%3 = add %1, %2

after:
%3 = constant_f32 2.0
```

Keeping `%3` means every later reference remains correct:

```text
%4 = multiply %0, %3
```

No operand remapping is needed during this pass.

The old `%1` and `%2` may now be unused, but removing them is a separate
responsibility for dead-code elimination.

This separation makes each pass easier to understand and test:

```text
constant folding changes how a live value is produced
DCE removes values that are no longer needed
```

### Non-finite results are not folded

The MiniIR `ConstantF32` operation requires a finite value.

Runtime arithmetic can still overflow:

```text
largest finite float × 2 → infinity
```

Replacing that operation with `constant_f32 infinity` would create invalid
IR.

The pass therefore leaves such arithmetic unchanged rather than:

```text
throwing
clamping the result
creating an illegal constant
```

This preserves the original runtime behavior.

## Dead-code elimination

Dead code is work whose result cannot affect the program output.

In the folded scale program:

```text
%5 = constant_f32 100.0
%6 = add %0, %5
return %4
```

`%6` is never used.

Therefore:

```text
%6 is dead
%5 is also dead because only %6 needed it
```

After constant folding, the old constants `%1` and `%2` are also dead:

```text
%3 is now directly constant_f32 2.0
```

## Backward liveness analysis

Dead-code elimination first asks:

> Which SSA values can influence the returned value?

It begins at the end:

```text
return %4
```

Therefore:

```text
%4 is live
```

The pass walks backward.

Because `%4` is live:

```text
%4 = multiply %0, %3
```

its operands must also be live:

```text
%0 is live
%3 is live
```

Because `%3` is now a constant, it has no operands.

Nothing live refers to:

```text
%1
%2
%5
%6
```

so those values are dead.

Conceptually:

```text
seed Return operand as live
        ↓
walk operations backward
        ↓
if an operation's result is live,
mark all of its operands live
        ↓
everything else is dead
```

### All current operations are pure

DCE is safe here because every current MiniIR operation is pure:

```text
Input
ConstantF32
Add
Multiply
ReLU
```

Pure means that executing an operation matters only through its returned
value.

There are currently no operations such as:

```text
printing
file writes
random-state updates
in-place buffer mutation
network requests
atomics with externally visible effects
```

In a richer compiler, an unused result does **not** automatically mean the
operation can be deleted.

Side-effecting operations must usually remain even when their result is
unused.

Lesson 23's DCE rule is valid specifically because all current operations are
side-effect-free.

## Why DCE renumbers SSA values

After deleting dead values, the old IDs would contain holes:

```text
%0 kept
%1 removed
%2 removed
%3 kept
%4 kept
```

But our verifier expects dense sequential IDs:

```text
%0, %1, %2
```

DCE therefore builds an old-to-new mapping:

```text
old %0 → new %0
old %1 → removed
old %2 → removed
old %3 → new %1
old %4 → new %2
old %5 → removed
old %6 → removed
```

Every kept operation receives its new result ID, and every operand is
rewritten through the same table.

The final optimized scale region becomes:

```text
%0 = input @x : tensor<5xf32>
%1 = constant_f32 2.0 : f32
%2 = multiply %0, %1 : tensor<5xf32>
return %2
```

The numeric IDs changed, but the calculation did not.

### Multi-use values remain live

Consider:

```text
%2 = multiply %0, %1
%3 = add %2, %2
```

`%2` is used twice.

Liveness is not a reference counter that deletes a value after one use.
Once `%2` is found to contribute to Return, it remains live, and both operand
references are remapped to its same new ID.

## The optimization report

For the scale stage, the report is:

```text
constant folding:
  folded values: 1
  %3 add -> constant_f32 2.0
dead-code elimination:
  removed values: 4
  remap:
    %0 -> %0
    %1 -> removed
    %2 -> removed
    %3 -> %1
    %4 -> %2
    %5 -> removed
    %6 -> removed
```

The report is metadata for understanding and testing the pass.

The optimized `Program` remains the object consumed by later compiler
stages.

## What elementwise fusion honestly means here

Lesson 22 received one `Program` already defined as one kernel region.

It directly translated that region into one MSL function:

```text
one Program → one Metal kernel
```

Lesson 22 did not inspect several regions and decide to fuse them.

Lesson 23 introduces a small level above Lesson 22:

```text
PipelineStage 0: scale Program
PipelineStage 1: shift Program
PipelineStage 2: activation Program
```

The fusion pass removes the boundaries between these compatible linear
elementwise regions.

That is real kernel-region fusion:

```text
three Programs that could each become a dispatch
        ↓
one Program that becomes one dispatch
```

## Describing the linear pipeline

A stage contains:

```cpp
struct PipelineStage {
    std::string name;
    Program program;
    std::optional<std::string> previous_output_input;
};
```

The demo constructs:

```cpp
std::vector<PipelineStage> stages{
    {
        "scale",
        optimized_scale,
        std::nullopt,
    },
    {
        "shift",
        shift,
        std::string("scaled"),
    },
    {
        "activation",
        activation,
        std::string("shifted"),
    },
};
```

The first stage has no previous output.

The shift stage says:

```text
my input @scaled receives the scale stage's output
```

The activation stage says:

```text
my input @shifted receives the shift stage's output
```

This lesson supports only a linear chain:

```text
stage 0 → stage 1 → stage 2
```

It does not yet represent an arbitrary graph.

## Edge substitution

Before fusion, the shift stage begins with:

```text
%0 = input @scaled : tensor<5xf32>
```

That input exists because the shift region would normally receive a real
intermediate buffer.

During fusion, no buffer is needed between the scale and shift calculations.

The pass substitutes:

```text
shift's @scaled input
        ↓
scale's returned SSA value
```

Likewise:

```text
activation's @shifted input
        ↓
shift's returned SSA value
```

The connected `Input` operations and intermediate `Return` operations
disappear.

The arithmetic operations are copied into one new Program while their SSA
operands are remapped.

Only genuine external inputs remain.

For this pipeline, the only external input is:

```text
@x
```

Lesson 23 requires external input names to be unique across stages.

Two stages may both spell an input `@bias`, but that does not prove the
unfused executions would receive the same buffer. Silently merging those
inputs could change the program's meaning.

Until a later graph layer provides explicit shared-input identity, repeated
external names are therefore rejected as ambiguous.

## The fused IR

The fusion pass produces:

```text
%0 = input @x : tensor<5xf32>
%1 = constant_f32 2.0 : f32
%2 = multiply %0, %1 : tensor<5xf32>
%3 = constant_f32 3.0 : f32
%4 = add %2, %3 : tensor<5xf32>
%5 = relu %4 : tensor<5xf32>
return %5
```

Notice what is absent:

```text
input @scaled
input @shifted
intermediate Return operations
```

The stage edges have become direct SSA dependencies:

```text
%2 ──→ %4 ──→ %5
```

## The fusion API

```cpp
FusionResult fuse_linear_elementwise_pipeline(
    const std::vector<PipelineStage>& stages
);
```

`FusionResult` contains:

```text
the fused Program
ordered stage names
materialized fusion edges
dispatch count before and after
number of intermediate buffers avoided
```

For this example:

```text
elementwise fusion:
  stages: scale -> shift -> activation
  edge: scale -> shift via @scaled
  edge: shift -> activation via @shifted
  dispatches: 3 -> 1
  intermediate buffers avoided: 2
```

The counts describe this particular linear execution plan.

They are not timing measurements.

## Fusion validation

`fuse_linear_elementwise_pipeline()` rejects malformed or unsafe pipelines.

It requires:

```text
at least one stage
nonempty, unique stage names
no previous-output edge on the first stage
a declared previous-output input on every later stage
the declared input must exist
the declared input must contribute to that stage's output
producer output and consumer input types must match exactly
the connected input must contribute to the consumer output
external input names must be unique across stages
every non-scalar fused value must use the final output shape
```

For example, this cannot be fused:

```text
producer output: tensor<5xf32>
consumer input:  tensor<4xf32>
```

The error is a `FusionError`.

Verification of each individual MiniIR `Program` still uses the existing
`VerificationError`.

## Before and after fusion

### Unfused

```text
x buffer
    ↓
scale MetalExecutable
    ↓ dispatch 1
scaled intermediate buffer
    ↓
shift MetalExecutable
    ↓ dispatch 2
shifted intermediate buffer
    ↓
activation MetalExecutable
    ↓ dispatch 3
output buffer
```

Counts:

```text
dispatches             = 3
intermediate buffers   = 2
```

### Fused

```text
x buffer
    ↓
one fused MetalExecutable
    ↓ dispatch 1
output buffer
```

Counts:

```text
dispatches             = 1
intermediate buffers   = 0
```

There are still lane-local intermediate values inside the fused MSL:

```metal
const float v2 = v0 * v1;
const float v4 = v2 + v3;
const float v5 = max(v4, 0.0f);
```

Those are not whole tensor buffers.

Each active lane owns its own scalar instances of those values.

## Why fewer dispatches may help

The fused path avoids:

```text
two command-encoding boundaries
two whole intermediate tensor buffers
writing scaled to a buffer and reading it back
writing shifted to a buffer and reading it back
```

However, fusion can also increase:

```text
generated kernel size
register pressure
compilation cost
complexity
```

Some operations cannot legally share a kernel at all.

Therefore the only claim made by Lesson 23 is:

```text
the demonstrated schedule uses fewer dispatches and temporary buffers
```

It does not claim:

```text
an exact speedup
that fusion is always profitable
that every operation can be fused
```

## Final lowering through Lesson 22

Fusion does not generate MSL itself.

It returns one ordinary typed MiniIR `Program`.

The existing backend remains responsible for Metal:

```text
FusionResult.program
        ↓ lower_to_metal()
MetalKernelPlan
        ↓ compile_metal()
MetalExecutable
        ↓ make_dispatch() + make_launch()
MiniGPU Queue
        ↓
Apple GPU
```

The fused Metal plan has only one input:

```text
buffer(0) input @x
buffer(1) output
buffer(2) element_count
```

Inputs named:

```text
@scaled
@shifted
```

do not appear in the final ABI because they became internal SSA edges.

The generated MSL contains multiplication, addition and ReLU in the same
kernel function.

## Unfused queue ordering

The demo submits the three unfused dispatches to one MiniGPU queue:

```text
submit scale
submit shift
submit activation
wait for scale
wait for shift
wait for activation
```

Commands on that Metal queue execute in order.

Therefore:

```text
scale fills scaled before shift reads it
shift fills shifted before activation reads it
```

All three commands are submitted before the CPU begins waiting, so these waits
do not insert new boundaries between the GPU submissions.

Waiting in submission order surfaces the earliest producer error. The final
activation wait then places the CPU after the entire pipeline.

The intermediate buffers are retained by the submitted dispatches and
events while the GPU needs them.

## CPU and GPU equivalence

The demo evaluates:

```text
optimized unfused Programs on the CPU
fused Program on the CPU
optimized unfused Programs on the GPU
fused Program on the GPU
```

All four paths produce:

```text
[0, 1, 3, 5, 7]
```

This checks two different kinds of equivalence:

```text
optimization preserved the original mathematical meaning
Metal lowering preserved the optimized MiniIR meaning
```

## Reusing the compiled fused executable

The demo compiles the fused kernel once and runs it again with different
buffers.

Second input:

```text
[0, 1, 2, 3, 4]
```

Second output:

```text
[3, 5, 7, 9, 11]
```

The compiled `MetalExecutable` contains the algorithm and ABI.

A dispatch supplies one particular set of input and output buffers.

## What the demo tests

### Constant-folding tests

```text
exact original IR
exact folded IR
1 + 1 becomes constant 2
nested Multiply, ReLU and Add constants fold forward
non-finite results are deliberately not folded
fold records contain value ID, old opcode and result
folding is deterministic
```

### Dead-code-elimination tests

```text
exact optimized IR
four dead values are removed from the scale program
old-to-new SSA remapping is exact
a live value used more than once remains live
the resulting Program passes verification
optimization is idempotent
the original Program is unchanged
original and optimized CPU results match directly
```

Idempotent means:

```text
optimizing an already optimized Program changes nothing
```

### Fusion tests

```text
exact fused IR
deterministic fused Program and report
three ordered stage names
two exact pipeline edges
3 dispatches become 1
2 intermediate buffers are avoided
fusing one already-fused stage changes nothing
empty and malformed pipelines are rejected
missing edge inputs are rejected
unused declared edge inputs are rejected
duplicate stage names are rejected
repeated external names are rejected as ambiguous
producer/consumer shape mismatches are rejected
```

### Metal and runtime tests

```text
the final Metal ABI contains only external input @x
intermediate inputs are absent from generated MSL
generated MSL contains Multiply, Add and ReLU
unfused real GPU output matches the CPU
fused real GPU output matches the CPU
unfused and fused GPU outputs match each other
instrumented dispatch and temporary-buffer counts match fusion metadata
all three unfused Events are waited, so producer errors are not ignored
the compiled fused executable is reusable with new buffers
```

The normal and ASan/UBSan builds test the host C++ and Objective-C++ code.

ASan and UBSan do not inspect memory accesses performed inside GPU MSL.
The generated bounds guard, Metal compilation and real GPU result checks
cover the GPU path used here.

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
  outputs/infer-lab/phase0/lesson23-optimization-fusion/optimizer.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson23-optimization-fusion/demo.mm \
  -framework Foundation \
  -framework Metal \
  -o work/lesson23_optimization_fusion

./work/lesson23_optimization_fusion
```

Metal GPU access may require running outside a restricted sandbox.

## Sanitized build

```bash
xcrun clang++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fobjc-arc \
  -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  outputs/infer-lab/phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  outputs/infer-lab/phase0/lesson22-metal-codegen/metal_backend.cpp \
  outputs/infer-lab/phase0/lesson23-optimization-fusion/optimizer.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson23-optimization-fusion/demo.mm \
  -framework Foundation \
  -framework Metal \
  -o work/lesson23_optimization_fusion_sanitized

./work/lesson23_optimization_fusion_sanitized
```

## Expected key output

```text
Lesson 23: optimization and linear elementwise fusion

Original scale IR:
%0 = input @x : tensor<5xf32>
%1 = constant_f32 1.0 : f32
%2 = constant_f32 1.0 : f32
%3 = add %1, %2 : f32
%4 = multiply %0, %3 : tensor<5xf32>
%5 = constant_f32 100.0 : f32
%6 = add %0, %5 : tensor<5xf32>
return %4

After constant folding:
%0 = input @x : tensor<5xf32>
%1 = constant_f32 1.0 : f32
%2 = constant_f32 1.0 : f32
%3 = constant_f32 2.0 : f32
%4 = multiply %0, %3 : tensor<5xf32>
%5 = constant_f32 100.0 : f32
%6 = add %0, %5 : tensor<5xf32>
return %4

After dead-code elimination:
%0 = input @x : tensor<5xf32>
%1 = constant_f32 2.0 : f32
%2 = multiply %0, %1 : tensor<5xf32>
return %2

Fused three-stage IR:
%0 = input @x : tensor<5xf32>
%1 = constant_f32 2.0 : f32
%2 = multiply %0, %1 : tensor<5xf32>
%3 = constant_f32 3.0 : f32
%4 = add %2, %3 : tensor<5xf32>
%5 = relu %4 : tensor<5xf32>
return %5

constant folding:
  folded values: 1
  %3 add -> constant_f32 2.0
dead-code elimination:
  removed values: 4
  remap:
    %0 -> %0
    %1 -> removed
    %2 -> removed
    %3 -> %1
    %4 -> %2
    %5 -> removed
    %6 -> removed

elementwise fusion:
  stages: scale -> shift -> activation
  edge: scale -> shift via @scaled
  edge: shift -> activation via @shifted
  dispatches: 3 -> 1
  intermediate buffers avoided: 2

Unfused GPU: 3 dispatches, 2 intermediate buffers, output [0, 1, 3, 5, 7]
Fused GPU:   1 dispatch, 0 intermediate buffers, output [0, 1, 3, 5, 7]
Reused fused executable output: [3, 5, 7, 9, 11]

Optimization, fusion, Metal equivalence and reuse checks: PASS
PASS: optimized unfused and fused programs agree on the real Apple GPU.
```

## Main takeaways

1. An analysis computes facts; a transformation pass produces rewritten IR.
2. Constant propagation walks forward because SSA operands are defined
   before their uses.
3. Constant folding replaces known scalar arithmetic with constants.
4. Folding keeps result IDs because it replaces operations without deleting
   values.
5. Liveness begins at Return and flows backward through dependencies.
6. DCE removes values that cannot affect Return.
7. DCE is currently safe because all MiniIR operations are pure.
8. DCE renumbers values because deletion leaves holes in dense SSA IDs.
9. A `Program` is one kernel region; a pipeline connects several regions.
10. Fusion substitutes inter-stage inputs with the preceding stage's SSA
    output.
11. Lesson 23's fusion genuinely changes three potential dispatches into one.
12. The fused result is an ordinary Program lowered by the existing Lesson 22
    backend.
13. Fewer dispatches and buffers do not automatically prove a speedup.
14. Compiler transformations must preserve observable results.

## What to understand versus skim

Understand:

```text
analysis versus transformation pass
forward constant table
operation replacement while retaining SSA ID
Return-seeded backward liveness
old-to-new SSA remapping
why purity permits deletion
Program versus PipelineStage
previous-output edge substitution
three kernel regions versus one fused region
dispatch count versus execution time
optimized Program → Lesson 22 → Metal
```

Skim:

```text
string formatting in reports
host vector-printing helpers
exception-test helper templates
Objective-C autorelease pool
buffer upload/download loops
boilerplate includes
```

## Deferred scope

Lesson 23 does not implement:

```text
arbitrary DAG or graph fusion
branching, cycles or multiple outputs
fusion across reductions, softmax or matmul
barrier-aware or side-effect-aware fusion
device cost models
register-pressure estimation
autotuning
performance benchmarking
common-subexpression elimination
algebraic rules such as x + 0 or x × 1
reassociation or fast-math transformations
in-place buffer reuse or general memory planning
kernel caching
dynamic shapes
general broadcasting
f16 Metal execution
MiniTensor operator syntax
```

Later lessons will need richer graph and tensor layers before fusion can make
production-style legality and profitability decisions.
