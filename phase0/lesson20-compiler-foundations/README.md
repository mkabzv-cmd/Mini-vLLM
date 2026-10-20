# Lesson 20 — First in-memory compiler IR

## Outcome

This lesson implements the first real compiler layer:

```text
explicit builder calls
        ↓
in-memory operations and values
        ↓
SSA-like IR
        ↓
structural verification
        ↓
deterministic text printer
        ↓
tiny CPU semantic check
```

It deliberately does not generate MSL or execute on the GPU yet.

The detailed theory discussed before implementation is preserved separately
in [LEARNING_GUIDE.md](./LEARNING_GUIDE.md).

## The example program

We want to represent:

```cpp
relu(a + b * 2.0f)
```

For:

```text
a = [-3,  1,  5, -2]
b = [ 1,  2, -4, 10]
```

the answer is:

```text
b × 2       = [ 2, 4, -8, 20]
a + b × 2   = [-1, 5, -3, 18]
relu(...)   = [ 0, 5,  0, 18]
```

Lesson 20 does not use the convenient future syntax:

```cpp
Tensor result = relu(a + b * 2.0f);
```

MiniTensor does not exist yet. We use the compiler's explicit builder:

```cpp
ProgramBuilder builder;

Value a = builder.input("a");
Value b = builder.input("b");
Value two = builder.constant_f32(2.0f);
Value scaled = builder.multiply(b, two);
Value added = builder.add(a, scaled);
Value result = builder.relu(added);

builder.return_value(result);
Program program = builder.build();
```

The builder produces:

```text
%0 = input @a
%1 = input @b
%2 = constant_f32 2.0
%3 = multiply %1, %2
%4 = add %0, %3
%5 = relu %4
return %5
```

## Builder records; it does not calculate

This call:

```cpp
Value scaled = builder.multiply(b, two);
```

does not multiply any tensor data.

It appends an operation:

```text
%3 = multiply %1, %2
```

and returns a `Value` handle referring to the future result `%3`.

The distinction is:

```text
Operation
    instruction describing work to perform

Value
    ID referring to an operation's future result

Program
    ordered collection of operations
```

The real input numbers arrive only when a backend executes or interprets the
program.

## What is stored in memory?

The IR is not initially written to a temporary `.ir` file.

The `Program` owns:

```cpp
std::vector<Operation> operations_;
```

Conceptually:

```text
operations_[0]
    opcode = Input
    result = %0
    name   = "a"

operations_[1]
    opcode = Input
    result = %1
    name   = "b"

operations_[2]
    opcode   = ConstantF32
    result   = %2
    constant = 2.0

operations_[3]
    opcode   = Multiply
    result   = %3
    operands = [%1, %2]
```

`format_program()` converts those objects into readable text for humans.

The text is a view of the in-memory program, not the program's fundamental
storage format.

## SSA-like value numbering

Each result receives one ID:

```text
%0
%1
%2
...
```

Each ID is defined exactly once:

```text
%3 = multiply %1, %2
```

No later operation replaces `%3`.

This makes dependencies explicit:

```text
%1 ──┐
     ├── multiply → %3
%2 ──┘

%0 ──┐
     ├── add → %4
%3 ──┘

%4 → relu → %5
```

This lesson calls the representation SSA-like because it uses
single-assignment values but does not yet implement branches, loops or phi
nodes.

## What an `Operation` contains

The core structure is:

```cpp
struct Operation {
    Opcode opcode;
    std::optional<ValueId> result;
    std::vector<ValueId> operands;
    std::string input_name;
    float constant_value;
};
```

Different opcodes use different fields:

| Opcode | Produces a value? | Operands | Extra data |
|---|---:|---:|---|
| `Input` | yes | 0 | input name |
| `ConstantF32` | yes | 0 | float value |
| `Multiply` | yes | 2 | none |
| `Add` | yes | 2 | none |
| `Relu` | yes | 1 | none |
| `Return` | no | 1 | none |

`Return` consumes `%5` but does not create `%6`.

## Why `Value` remembers its builder

A numeric ID alone is not enough:

```text
builder A has a %0
builder B also has a %0
```

Those values belong to different programs.

`Value` carries a hidden builder identity. This is rejected:

```cpp
Value from_a = builder_a.input("a");
Value from_b = builder_b.input("b");

builder_b.add(from_a, from_b); // error
```

Otherwise the same numeric `%0` could accidentally refer to unrelated
operations.

The identity is compiler metadata. It is not tensor data.

## Structural verification

`verify()` checks the structure of the program:

```text
every opcode has the correct number of operands
every operand was defined earlier
result IDs are sequential and unique
inputs have nonempty, unique names
input names use simple identifier syntax
every value-producing operation has a result
return produces no value
return is the final operation
the program has a return
constants are finite
opcodes do not carry attributes that belong to another opcode
```

For example:

```text
%1 = add %0, %99
```

is rejected because `%99` was not defined before it was used.

This is only structural verification.

Lesson 21 will introduce richer types, tensor shapes and diagnostics such as:

```text
f32 versus f16
scalar versus tensor
shape [4] versus shape [8]
broadcasting rules
```

## Deterministic printing

The same program must always print the same text:

```text
same operation order
same value numbers
same constant formatting
same final newline
```

Determinism matters for:

```text
tests
debugging
compiler logs
cache keys
comparing IR before and after a pass
```

The demo formats the program twice and requires byte-identical results.

## The CPU evaluator

`evaluate_element()` is a tiny temporary interpreter:

```cpp
float answer = evaluate_element(
    program,
    {
        {"a", -3.0f},
        {"b", 1.0f},
    }
);
```

It walks through the operations:

```text
%0 = input a        → -3
%1 = input b        → 1
%2 = constant       → 2
%3 = multiply       → 1 × 2 = 2
%4 = add            → -3 + 2 = -1
%5 = relu           → 0
return %5           → 0
```

The demo calls it once per vector element and obtains:

```text
[0, 5, 0, 18]
```

This evaluator is not our final inference runtime and is not a GPU compiler.

Its purposes are:

```text
prove that the IR means what we think it means
provide a simple correctness reference
show that one source IR can support different execution paths
```

Later:

```text
CPU evaluator
    interprets the IR directly

lowering plus Metal backend
    derives a lower representation and MSL from the same source meaning
```

## Why there are no tensor types yet

Every runtime binding in this lesson is one `float`.

The demo applies the program separately to each element:

```cpp
for each index:
    evaluate_element(
        program,
        {
            {"a", a[index]},
            {"b", b[index]},
        }
    );
```

This matches the future elementwise GPU model:

```text
one GPU lane evaluates the scalar expression for one index
```

Types, tensor shapes and basic broadcasting are deliberately deferred to
Lesson 21 so they can be learned separately. Strides and richer layout
questions come later.

## Architecture

```text
demo.cpp
    builds and exercises one program
        ↓
miniir.hpp
    public IR objects and builder API
        ↓
miniir.cpp
    builder, verifier, printer and evaluator
```

The long conceptual reference is:

```text
LEARNING_GUIDE.md
```

## What the demo checks

The executable checks:

```text
exact expected IR text
repeated printing is deterministic
evaluation returns [0, 5, 0, 18]
undefined operands are rejected
wrong operand counts are rejected
missing return is rejected
duplicate input names are rejected
invalid input names are rejected
attributes on the wrong opcode are rejected
cross-builder values are rejected
missing runtime input bindings are rejected
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
  outputs/infer-lab/phase0/lesson20-compiler-foundations/miniir.cpp \
  outputs/infer-lab/phase0/lesson20-compiler-foundations/demo.cpp \
  -o work/lesson20_first_ir

./work/lesson20_first_ir
```

No Metal framework is needed because this lesson stops before MSL generation
and GPU execution.

## Expected output

```text
Lesson 20: first in-memory compiler IR

Builder recorded this program:
%0 = input @a
%1 = input @b
%2 = constant_f32 2.0
%3 = multiply %1, %2
%4 = add %0, %3
%5 = relu %4
return %5

The CPU evaluator applies the same IR to each element:
a      = [-3, 1, 5, -2]
b      = [1, 2, -4, 10]
result = [0, 5, 0, 18]

Structural and error-path checks: PASS
PASS: builder, SSA-like IR, verifier, printer and evaluator succeeded.
```

## Main takeaways

1. The builder records operations; it does not calculate tensor values.
2. A `Value` is a handle to a future result, not the result's data.
3. An `Operation` contains an opcode, operands and optional attributes.
4. A `Program` is an ordered in-memory collection of operations.
5. Every SSA-like result ID is defined once.
6. Operands refer to previously defined result IDs.
7. Verification checks program structure before another backend consumes it.
8. Printing makes invisible in-memory IR understandable.
9. The CPU evaluator proves semantics but is not the final backend.
10. Lesson 20 has one IR level only.

## What to understand versus skim

Understand:

```text
builder call → Operation
Operation result → Value ID
Value ID → dependency edge
vector<Operation> → in-memory Program
verify before consuming IR
printer displays but does not store the IR
evaluator interprets but does not compile the IR
```

Skim:

```text
exception helper templates
vector printing
float formatting details
negative-test construction
boilerplate includes
```

## What comes next

Lesson 21 will add:

```text
types
tensor shapes
type and shape inference
scalar-to-tensor broadcasting rules
richer verifier diagnostics
```

Lesson 22 will add:

```text
MSL generation
buffer-binding metadata
launch metadata
MiniGPU execution
```

Lesson 23 will add:

```text
constant folding
dead-code elimination
elementwise fusion
```

Lesson 24 will add:

```text
MiniTensor
operator overloads
captured expressions
evaluate(...)
```
