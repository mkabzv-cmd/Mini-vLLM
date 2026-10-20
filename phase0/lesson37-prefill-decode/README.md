# Lesson 37 — Prefill versus decode

## Result

Lesson 36 proved that our model can generate tokens:

```text
complete sequence
       ↓
full model forward
       ↓
choose one new token
       ↓
append it
       ↓
repeat
```

That implementation is correct, but it repeatedly recalculates old tokens.

Lesson 37 gives names and exact shapes to the two phases that a cache-aware
inference engine will eventually use:

```text
prompt
   ↓
prefill
   ↓
first generated token
   ↓
decode one token at a time
```

This lesson does **not** implement a KV cache yet.

It builds a small CPU-side plan that answers:

```text
Which phase are we in?
How many new token positions enter the model?
What position does the new token occupy?
How many earlier tokens must attention see?
How much work is Lesson 36 repeating?
How much work could a KV cache avoid?
```

The distinction is important:

```text
current Lesson 36 runtime
    still recomputes the complete prefix

Lesson 37 cache-aware plan
    describes the work a future runtime should perform

Lessons 38–39
    will create the cache and perform that work for real
```

## First clarification: two meanings of “decode”

The word `decode` appears in two unrelated places.

### Tokenizer decoding

A tokenizer can convert token IDs back into text:

```text
[1, 15043] → "Hello"
```

Python may call this:

```python
tokenizer.decode(token_ids)
```

### Model decode phase

An inference engine uses `decode` to mean:

```text
process the newest token
while reusing information from earlier tokens
to predict the following token
```

Lesson 37 is about the **model decode phase**, not converting token IDs into
text.

## The most important timing detail

The model consumes one token and predicts the token that comes **after** it.

Suppose the prompt is:

```text
A B
```

Prefill consumes:

```text
A B
```

and predicts:

```text
C
```

After we select `C`, a decode call can consume:

```text
C
```

and predict:

```text
D
```

Then the next decode call consumes `D` and predicts `E`.

```text
prefill(A B) → C
decode(C)    → D
decode(D)    → E
```

The token entering a decode call is not the token that call is about to
generate. It is the previously generated token.

## Phase 1: prefill

Prefill processes all prompt tokens together.

For a prompt of `P` tokens:

```text
input tokens = P
```

Suppose:

```text
P = 4

prompt = A B C D
```

The model processes:

```text
positions 0, 1, 2, 3
```

It produces logits for every prompt position, but generation uses only the
final row:

```text
logits after D → scores for the first generated token
```

Therefore the first generated token comes from **prefill**.

It does not require a decode call.

### Prefill attention shapes

Let:

```text
P = prompt token count
H = number of attention heads
D = head dimension
```

For one transformer layer:

```text
Q [H, P, D]
K [H, P, D]
V [H, P, D]
```

The attention scores are:

```text
Q @ Kᵀ

[H, P, D] @ [H, D, P]
        ↓
[H, P, P]
```

There is one query row for every prompt token and one key column for every
prompt token.

The causal mask prevents earlier prompt tokens from looking into the future:

```text
        keys
        A  B  C  D

query A ✓  ✗  ✗  ✗
query B ✓  ✓  ✗  ✗
query C ✓  ✓  ✓  ✗
query D ✓  ✓  ✓  ✓
```

The context result has shape:

```text
[H, P, D]
```

Every prompt query gets one context vector.

### What a future KV cache will keep

During prefill, every layer calculates key and value vectors for all prompt
positions:

```text
K for positions 0 ... P-1
V for positions 0 ... P-1
```

A future cache-aware runtime will preserve those tensors instead of
recalculating them before every new token.

Lesson 37 only describes that future behavior. It does not store them yet.

## Phase 2: one-token model decode

After prefill selects the first generated token, generation continues one
token at a time.

Suppose the prompt contains:

```text
A B C D
```

and prefill selects:

```text
E
```

The future decode call will process only `E`:

```text
decode input = E
```

But `E` must be able to attend to:

```text
A B C D E
```

The earlier keys and values will come from the KV cache. Only the key and
value for `E` need to be newly calculated.

### Decode attention shapes

Let:

```text
C = number of cached tokens before this decode call
H = number of attention heads
D = head dimension
```

The new token produces:

```text
Q_new [H, 1, D]
K_new [H, 1, D]
V_new [H, 1, D]
```

Before adding the new token, the future cache contains:

```text
K_cache [H, C, D]
V_cache [H, C, D]
```

After adding the new key and value, attention can see:

```text
K_all [H, C + 1, D]
V_all [H, C + 1, D]
```

The score calculation is:

```text
Q_new @ K_allᵀ

[H, 1, D] @ [H, D, C + 1]
         ↓
[H, 1, C + 1]
```

There is:

```text
one query row
one score for every past token
one score for the current token itself
```

The context result is:

```text
[H, 1, C + 1] @ [H, C + 1, D]
                    ↓
               [H, 1, D]
```

Only the newest token needs a new context vector. Earlier tokens already
received theirs during prefill or an earlier decode call.

### Why one-token decode usually needs no explicit causal mask

The cache contains only:

```text
past tokens + current token
```

No future tokens exist yet.

Therefore the single new query cannot accidentally attend to a future token.
Causality is enforced by which positions are present.

Prefill still needs a causal mask because multiple prompt positions are
processed together.

## `start_position` during the two phases

RoPE needs each token's position in the complete sequence.

During prefill:

```text
start_position = 0
```

For a four-token prompt:

```text
local token 0 → sequence position 0
local token 1 → sequence position 1
local token 2 → sequence position 2
local token 3 → sequence position 3
```

During one-token decode, the small input tensor always has local token index
zero. Its complete-sequence position comes from the number of tokens already
cached:

```text
start_position = cached_tokens_before
```

If four prompt tokens are cached:

```text
decode tensor's local token index = 0
start_position                   = 4
RoPE position                    = 4 + 0 = 4
```

For the following decode call:

```text
cached_tokens_before = 5
RoPE position        = 5
```

## Exact example: `P = 2`, `G = 8`

Lesson 36 uses:

```text
prompt text = "Hello"
prompt IDs  = [1, 15043]
P           = 2 prompt tokens
G           = 8 requested generated tokens
```

Call the generated tokens:

```text
g1 g2 g3 g4 g5 g6 g7 g8
```

For the bundled random model they are:

```text
g1 = 1789
g2 = 19053
g3 = 25470
g4 = 2323
g5 = 21742
g6 = 13486
g7 = 13934
g8 = 26715
```

The exact logical timeline is:

| Prediction | Future phase | Tokens entering future model call | `start_position` | Cached before | Attention scores per head | Produces |
|---:|---|---:|---:|---:|---:|---|
| 1 | prefill | 2 prompt tokens | 0 | 0 | `[2,2]` | `g1` |
| 2 | decode | `g1` only | 2 | 2 | `[1,3]` | `g2` |
| 3 | decode | `g2` only | 3 | 3 | `[1,4]` | `g3` |
| 4 | decode | `g3` only | 4 | 4 | `[1,5]` | `g4` |
| 5 | decode | `g4` only | 5 | 5 | `[1,6]` | `g5` |
| 6 | decode | `g5` only | 6 | 6 | `[1,7]` | `g6` |
| 7 | decode | `g6` only | 7 | 7 | `[1,8]` | `g7` |
| 8 | decode | `g7` only | 8 | 8 | `[1,9]` | `g8` |

Notice the final row:

```text
the model processes g7
        ↓
the model predicts g8
        ↓
generation stops at the configured eight-token limit
```

`g8` is returned to the caller, but it does not need to enter the model
unless we want to generate `g9`.

## Why eight generated tokens need only seven decode calls

Prefill already produces the first generated token:

```text
prefill → g1
```

The remaining seven tokens require seven decode calls:

```text
decode(g1) → g2
decode(g2) → g3
decode(g3) → g4
decode(g4) → g5
decode(g5) → g6
decode(g6) → g7
decode(g7) → g8
```

Therefore, when `G > 0`:

```text
prefill calls = 1
decode calls  = G - 1
total model calls = G
```

For `G = 8`:

```text
prefill calls = 1
decode calls  = 7
total calls   = 8
```

If we later ask for one more token, then `g8` enters another decode call and
that call predicts `g9`.

## Token-position work: `44` versus `9`

### What Lesson 36 actually does now

Lesson 36 has no cache. It reruns the complete prefix for every prediction:

| Prediction | Complete prefix recomputed |
|---:|---:|
| `g1` | 2 tokens |
| `g2` | 3 tokens |
| `g3` | 4 tokens |
| `g4` | 5 tokens |
| `g5` | 6 tokens |
| `g6` | 7 tokens |
| `g7` | 8 tokens |
| `g8` | 9 tokens |

Total model-input token positions:

```text
2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 44
```

### What future cache-aware execution should do

A cache-aware implementation can process:

```text
prefill:  2 prompt tokens
decode:   1 token
decode:   1 token
decode:   1 token
decode:   1 token
decode:   1 token
decode:   1 token
decode:   1 token
```

Total:

```text
2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 9
```

So the comparison is:

```text
Lesson 36 actual full recomputation: 44 token positions
future cache-aware plan:              9 token positions
```

The value `9` is not a performance result achieved by Lesson 37. It is the
amount of model-input work that Lessons 38–39 are preparing us to perform.

## General token-position formulas

Let:

```text
P = number of prompt tokens
G = number of generated tokens requested
```

If `G > 0`, Lesson 36 processes prefix lengths:

```text
P, P+1, P+2, ... P+G-1
```

Therefore:

```text
full_recomputation
    = sum from i=0 to G-1 of (P + i)
    = G×P + G×(G-1)/2
```

Future cache-aware execution processes:

```text
P prompt tokens
+ one token for each of the remaining G-1 predictions
```

Therefore:

```text
cache_aware
    = P + (G - 1)
```

For `P = 2` and `G = 8`:

```text
full_recomputation
    = 8×2 + 8×7/2
    = 16 + 28
    = 44

cache_aware
    = 2 + 7
    = 9
```

If `G = 0`, no model call is needed and both totals are zero.

## Attention-score work: `284` versus `46`

Counting input token positions does not show all attention work.

For one attention head in Lesson 36, every full-prefix pass creates a square
score matrix:

```text
sequence length T → score shape [T,T] → T² score slots
```

For sequence lengths `2` through `9`:

```text
2² + 3² + 4² + 5² + 6² + 7² + 8² + 9²

= 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81

= 284 score slots per head per layer
```

A future cache-aware execution uses:

```text
prefill: [2,2] → 4 score slots

decode g1: [1,3] → 3
decode g2: [1,4] → 4
decode g3: [1,5] → 5
decode g4: [1,6] → 6
decode g5: [1,7] → 7
decode g6: [1,8] → 8
decode g7: [1,9] → 9
```

Total:

```text
4 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 46
```

So:

```text
Lesson 36 actual full-prefix score slots: 284 per head per layer
future cache-aware score slots:             46 per head per layer
```

This count describes allocated score-matrix entries. During prefill, the
causal mask makes some upper-triangle entries unusable, but our current
attention implementation still constructs the square matrix.

For a complete model, these counts are repeated across attention heads and
transformer layers.

### General attention-score formulas

Lesson 36 full recomputation:

```text
sum from i=0 to G-1 of (P + i)²
```

Future cache-aware execution:

```text
P²
+
sum from i=1 to G-1 of (P + i)
```

The first term is square because prefill contains `P` queries against `P`
keys.

Every later term is linear because decode contains one query against the
increasing key history.

You do not need to memorize these formulas. Remember the shapes:

```text
full recomputation: [T,T]
cached decode:      [1,T]
```

## A KV cache does not make attention constant

It is tempting to think:

```text
decode processes one token
therefore every decode step has constant work
```

That is not correct.

Only one new query is calculated, but that query still examines every
available key:

```text
early decode: [1,10]
later decode: [1,100]
much later:   [1,1000]
```

The KV cache avoids recreating old keys, values, hidden states and
projections. It does not remove the need for the newest query to attend to
history.

This distinction will matter when we later study:

```text
long-context inference
KV-cache memory use
FlashAttention-style kernels
paged attention
continuous batching
```

## Why prefill and decode perform differently

Although both phases execute the same model weights, their workloads have
very different shapes.

### Prefill

```text
many input tokens
many query rows
large matrix multiplications
high parallelism
```

Prefill is commonly more compute-heavy. The GPU has many token rows to work
on simultaneously.

### Decode

```text
one new input token per sequence
one query row
increasingly large KV-cache reads
narrow matrix operations
latency matters for every emitted token
```

Decode often becomes sensitive to memory bandwidth and launch overhead.
Large model weights and cached keys and values must be read to produce only
one new token per sequence.

This is why production inference engines frequently use different kernels,
batching strategies and scheduling decisions for the two phases.

## What the Lesson 37 code demonstrates

The code in this lesson is deliberately a planner, not a cache.

Given:

```text
prompt token count P
actual prediction count G
```

it creates one record for every prediction:

```text
phase
prediction number
new model-input token count
start position
cached tokens before the call
attention query count
score-matrix key-axis count
score slots per head per layer
```

It also calculates:

```text
Lesson 36 full-recomputation token positions
future cache-aware token positions

Lesson 36 full-recomputation attention-score slots
future cache-aware attention-score slots
```

The planner lets us test the semantics before introducing mutable GPU cache
storage.

It must not be confused with an optimization:

```text
planning cache-aware work ≠ executing cache-aware work
```

Lesson 36 remains the executable generation baseline.

## Build

Run from the workspace root:

### Build the small CPU-only planner tests

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
  outputs/infer-lab/phase0/lesson37-prefill-decode/phase_plan.cpp \
  outputs/infer-lab/phase0/lesson37-prefill-decode/tests.cpp \
  -o outputs/infer-lab/phase0/work/lesson37_phase_tests
```

The planner itself is ordinary CPU C++, so this small test executable does
not need Metal.

Run it:

```bash
outputs/infer-lab/phase0/work/lesson37_phase_tests
```

### Build the real-model demonstration

The demonstration reuses the complete Lesson 36 generator and the real Metal
model runtime:

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
  outputs/infer-lab/phase0/lesson31-linear-operator/linear.cpp \
  outputs/infer-lab/phase0/lesson32-batched-matmul/batched_matmul.cpp \
  outputs/infer-lab/phase0/lesson33-transformer-ops/transformer_ops.cpp \
  outputs/infer-lab/phase0/lesson34-attention-transformer-block/transformer_block.cpp \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/hf_loader.cpp \
  outputs/infer-lab/phase0/lesson35-huggingface-llama/hf_llama.cpp \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/generation.cpp \
  outputs/infer-lab/phase0/lesson37-prefill-decode/phase_plan.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson37-prefill-decode/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o outputs/infer-lab/phase0/work/lesson37_prefill_decode
```

## Run

Use the Lesson 36 tokenizer bridge with the Lesson 37 executable:

```bash
python3 \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/run_prompt.py \
  outputs/infer-lab/models/hf-tiny-random-llama \
  outputs/infer-lab/phase0/work/lesson37_prefill_decode \
  "Hello" \
  --max-new-tokens 8
```

For this prompt, the tokenizer supplies two prompt tokens and the
demonstration generates eight more:

```text
P = 2
G = 8
```

and prints the prefill/decode timeline together with:

```text
token positions:                   44 versus 9
score slots per head per layer:   284 versus 46
```

## What to read

You only need to understand:

1. The small phase and step structures in the public header.
2. The loop that creates one prefill step and `G-1` decode steps.
3. The checked formulas for the work totals.
4. The example table printed by `demo.cpp`.

The overflow checks are correctness protections. You can skim them.

## Main takeaways

1. Prefill processes the complete prompt.
2. The final prompt-position logits produce the first generated token.
3. Model decode is different from tokenizer decoding.
4. A decode call consumes the previously selected token and predicts the
   following token.
5. Generating `G` tokens needs one prefill call and only `G-1` decode calls.
6. Prefill attention has shape `[H,P,P]`.
7. One-token decode attention has shape `[H,1,C+1]`.
8. The decode token's RoPE `start_position` equals the number of tokens
   already cached.
9. A KV cache avoids recalculating earlier K/V and hidden-state work.
10. The newest query must still read and attend to the growing history.
11. Lesson 37 describes future cached execution; it does not implement it.

## What comes next

Lesson 37 answered:

```text
WHEN do prefill and decode happen?
WHAT are their shapes?
HOW MUCH repeated work should caching remove?
```

Lesson 38 will answer:

```text
WHERE are keys and values stored?
What is the per-layer cache layout?
How much capacity is allocated?
How are positions appended safely?
Who owns the cache and how long does it live?
```

Lesson 39 will then make the model execute the split for real:

```text
prefill(prompt, cache)
        ↓ first token

decode_one(previous_token, cache)
        ↓ following token
```

Its most important correctness test will be:

```text
cached generation token IDs
        ==
Lesson 36 full-recomputation token IDs
```

That is the transition from a correct model runner toward a real inference
engine.
