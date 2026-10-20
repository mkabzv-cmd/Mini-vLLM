# Lesson 21 — Types, shapes and scalar broadcasting

## Outcome

Lesson 20 could represent this calculation:

```text
relu(a + b * 2)
```

but its values did not say whether they represented:

```text
a scalar
a vector
a matrix-shaped tensor
f16 elements
f32 elements
```

Lesson 21 gives every SSA value an immutable type:

```text
type = dtype + shape
```

The compiler can now:

```text
infer the type produced by each operation
accept legal expressions
reject incompatible dtypes
reject incompatible tensor shapes
model scalar-to-tensor broadcasting
print typed IR
check stored type annotations
```

This remains a compiler lesson. It does not add Metal, GPU buffers,
optimization or MiniTensor.

## The typed example

The builder now declares input types:

```cpp
const Type vector4 =
    Type::tensor(DType::F32, {4});

ProgramBuilder builder;

Value a = builder.input("a", vector4);
Value b = builder.input("b", vector4);
Value two = builder.constant_f32(2.0f);
Value scaled = builder.multiply(b, two);
Value added = builder.add(a, scaled);
Value result = builder.relu(added);

builder.return_value(result);
Program program = builder.build();
```

The resulting IR is:

```text
%0 = input @a : tensor<4xf32>
%1 = input @b : tensor<4xf32>
%2 = constant_f32 2.0 : f32
%3 = multiply %1, %2 : tensor<4xf32>
%4 = add %0, %3 : tensor<4xf32>
%5 = relu %4 : tensor<4xf32>
return %5
```

The user declares only the input types.

The compiler derives:

```text
%2 → f32
%3 → tensor<4xf32>
%4 → tensor<4xf32>
%5 → tensor<4xf32>
```

## Type does not contain tensor data

A `Type` is compiler metadata.

For:

```text
tensor<4xf32>
```

it records:

```text
dtype = f32
shape = [4]
rank = 1
element count = 4
```

It does not contain the four numbers.

The separation is:

```text
Type
    describes what a value will look like

Value
    names a future result such as %3

runtime buffer
    will eventually hold the actual numbers
```

Lesson 21 has no runtime tensor-storage system yet.

## Dtype

Dtype describes how one element is represented.

This lesson knows:

```text
f16
f32
```

It does not perform implicit conversion:

```text
tensor<4xf16> + tensor<4xf32> → error
```

The user or a future compiler pass would need an explicit cast operation.
Cast operations are not part of this lesson.

`f16` is compiler metadata in Lesson 21. The small CPU evaluator deliberately
executes only `f32`; it does not pretend ordinary C++ `float` arithmetic has
real half-precision behavior.

## Shape

Shape describes the logical dimensions:

```text
f32                  shape []       scalar
tensor<4xf32>        shape [4]      rank-one tensor
tensor<2x3xf32>      shape [2, 3]   rank-two tensor
```

An empty shape means a scalar:

```text
f32
```

A one-element tensor is not a scalar:

```text
f32                 scalar, rank 0
tensor<1xf32>       tensor, rank 1
```

That distinction matters for broadcasting.

All explicit tensor dimensions in this lesson must be positive:

```text
tensor<4xf32>       valid
tensor<2x3xf32>     valid
tensor<0xf32>       rejected
```

Zero-sized, dynamic and symbolic dimensions are deferred.

## Type and shape inference

`Input` is declared by the user:

```text
%0 = input @a : tensor<4xf32>
```

`constant_f32` always produces a scalar `f32`:

```text
%2 = constant_f32 2.0 : f32
```

`ReLU` preserves the input type:

```text
relu tensor<4xf32> → tensor<4xf32>
relu f16           → f16
```

`Add` and `Multiply` use the same elementwise inference rules.

| Left | Right | Result |
|---|---|---|
| `f32` | `f32` | `f32` |
| `tensor<4xf32>` | `tensor<4xf32>` | `tensor<4xf32>` |
| `f32` | `tensor<4xf32>` | `tensor<4xf32>` |
| `tensor<4xf32>` | `f32` | `tensor<4xf32>` |
| `tensor<4xf32>` | `tensor<3xf32>` | error |
| `tensor<4xf16>` | `tensor<4xf32>` | error |

The rule is:

```text
1. Dtypes must match.
2. Equal shapes preserve that shape.
3. Scalar with tensor produces the tensor shape.
4. Otherwise, reject the operation.
```

## What scalar broadcasting means

In:

```text
tensor<4xf32> × f32
```

the scalar is logically used for every tensor element:

```text
[1, 2, 3, 4] × 2

means:

[1×2, 2×2, 3×2, 4×2]
```

The inferred result type is:

```text
tensor<4xf32>
```

The compiler does not create a physical tensor:

```text
[2, 2, 2, 2]
```

Broadcasting is initially a rule about meaning and result shape. A future
backend will generate code that reads the same scalar for every output
element.

## Broadcasting is deliberately limited

Lesson 21 supports only scalar broadcasting.

These are valid:

```text
f32 + tensor<4xf32>
tensor<4xf32> + f32
tensor<2x3xf32> × f32
```

These are rejected:

```text
tensor<1xf32> + tensor<4xf32>
tensor<2x3xf32> + tensor<3xf32>
tensor<2x1xf32> + tensor<2x3xf32>
```

Although libraries such as NumPy may support some of those expressions,
general dimension-by-dimension broadcasting requires more indexing and
layout rules. We add it later.

## How the builder infers a result

Every `Value` now carries:

```text
Value ID
declared or inferred Type
builder identity
```

Suppose:

```text
b.type()   = tensor<4xf32>
two.type() = f32
```

Then:

```cpp
Value scaled = builder.multiply(b, two);
```

performs compiler work:

```text
check f32 against f32
see tensor [4] against scalar []
apply scalar-broadcasting rule
infer tensor<4xf32>
record the Multiply operation
return Value(%3, tensor<4xf32>)
```

It still performs no numerical multiplication.

## Why store the inferred type in the operation?

The operation contains its result annotation:

```text
%3 = multiply %1, %2 : tensor<4xf32>
```

Later compiler stages can ask:

```text
What dtype does %3 have?
What is its rank?
How many logical elements does it have?
Can another operation legally consume it?
```

The stored annotation also gives the verifier something to check.

## The verifier recomputes types

While walking the operations, the verifier builds a type table:

```text
known_types[%0] = tensor<4xf32>
known_types[%1] = tensor<4xf32>
known_types[%2] = f32
```

For:

```text
%3 = multiply %1, %2 : tensor<4xf32>
```

it:

```text
reads the types of %1 and %2
repeats the inference rules
derives tensor<4xf32>
compares that with the stored annotation
```

Suppose a future compiler pass accidentally creates:

```text
%3 = multiply %1, %2 : f32
```

The verifier reports:

```text
declared result type f32 does not match
inferred type tensor<4xf32>
```

This is why the builder inferring a type does not make verification
unnecessary. Compiler transformations can introduce mistakes after initial
construction.

## Richer diagnostics

A dtype error reports both sides:

```text
add dtype mismatch:
left is tensor<4xf32>
right is tensor<4xf16>
implicit casts are not supported
```

A shape error reports the incompatible types and the supported rule:

```text
multiply shape mismatch:
left is tensor<4xf32>
right is tensor<3xf32>
Lesson 21 supports only equal tensor shapes or scalar broadcasting
```

The verifier also preserves Lesson 20's structural checks:

```text
operand arity
use before definition
sequential SSA IDs
valid input names
final Return
opcode attributes
```

## The CPU evaluator

`evaluate_f32_element()` still evaluates one logical output element.

For index zero:

```text
a[0] = -3
b[0] = 1
```

it evaluates:

```text
%0 = -3
%1 = 1
%2 = 2
%3 = 1 × 2 = 2
%4 = -3 + 2 = -1
%5 = relu(-1) = 0
```

The demo calls it once for every element:

```text
index 0 → 0
index 1 → 5
index 2 → 0
index 3 → 18
```

This evaluator does not allocate tensors and does not implement a GPU
backend. It is still a small semantic reference.

Shapes and broadcasting are compiler semantics here. The evaluator does not
bind or validate whole tensors. The caller supplies the appropriate element
from each tensor input for the logical output index.

If an input itself is scalar and the output is a tensor, the caller must pass
that same scalar binding on every invocation. A later whole-tensor runtime
will enforce this automatically.

## Architecture

```text
ProgramBuilder
    declares input types
    infers operation result types
        ↓
Program
    owns typed Operations
        ↓
verify()
    recomputes and checks types
        ↓
format_program()
    prints deterministic typed IR
        ↓
evaluate_f32_element()
    checks one element's numerical meaning
```

## What the demo checks

The executable tests:

```text
scalar metadata
rank-one and rank-two tensor metadata
element counts
f16 and f32 printing
tensor with scalar inference
scalar with tensor inference
scalar with scalar inference
equal tensor-shape inference
ReLU type preservation
f16 type propagation
dtype mismatch rejection
tensor-shape mismatch rejection
tensor<1xf32> is not a scalar
rank mismatch rejection
zero-dimension rejection
incorrect stored result-type rejection
missing result-type rejection
typed Return rejection
undefined SSA operand rejection
cross-builder Value rejection
f16 CPU-evaluation rejection
deterministic IR printing
f32 result [0, 5, 0, 18]
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
  outputs/infer-lab/phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  outputs/infer-lab/phase0/lesson21-types-shapes-broadcasting/demo.cpp \
  -o work/lesson21_typed_ir

./work/lesson21_typed_ir
```

## Expected output

```text
Lesson 21: types, shapes and scalar broadcasting

Typed IR:
%0 = input @a : tensor<4xf32>
%1 = input @b : tensor<4xf32>
%2 = constant_f32 2.0 : f32
%3 = multiply %1, %2 : tensor<4xf32>
%4 = add %0, %3 : tensor<4xf32>
%5 = relu %4 : tensor<4xf32>
return %5

The compiler inferred %3, %4 and %5 as tensor<4xf32>.
The scalar 2.0 participates in every tensor element without becoming a tensor copy.

a      = [-3, 1, 5, -2]
b      = [1, 2, -4, 10]
result = [0, 5, 0, 18]

Type, shape, broadcasting and diagnostic checks: PASS
PASS: Lesson 21 typed IR succeeded.
```

## Main takeaways

1. Every SSA value now has an immutable dtype and shape.
2. Inputs declare types; later operation types are inferred.
3. A scalar has shape `[]`; `tensor<1xf32>` is not a scalar.
4. Add and Multiply require identical dtypes.
5. Equal tensor shapes are compatible.
6. A scalar can broadcast to a tensor's shape.
7. General multidimensional broadcasting is not implemented yet.
8. Broadcasting does not require physically copying the scalar.
9. The verifier independently recomputes stored result types.
10. Types describe data but do not store the data.

## What to understand versus skim

Understand:

```text
Type = dtype + shape
scalar shape [] versus tensor shape [1]
input types are declared
operation result types are inferred
scalar broadcasting chooses the tensor result shape
dtype mismatch versus shape mismatch
verifier recomputes inferred types
type metadata versus runtime tensor data
```

Skim:

```text
float-print formatting
input-name parsing
exception helper templates
negative-test construction
shared_ptr builder identity
boilerplate includes
```

## What remains deferred

Lesson 21 does not implement:

```text
MSL generation or Metal execution
buffer bindings or launch geometry
general NumPy-style broadcasting
dynamic or symbolic dimensions
dtype promotion or casts
real f16 arithmetic
strides, layouts, views or transposes
tensor storage and allocation
optimization or fusion
matmul, reductions or softmax
MiniTensor operator syntax
```

Lesson 22 will lower this typed elementwise meaning toward MSL and execute
the generated kernel through MiniGPU.
