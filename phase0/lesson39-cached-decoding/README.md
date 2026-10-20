# Lesson 39 — Real cached Llama decoding

## Result

Lesson 38 gave us GPU storage for keys and values.

Lesson 39 makes the Llama model actually use that storage:

```text
prompt
  ↓
prefill the model and fill every layer's KV cache
  ↓
select the first generated token
  ↓
send only that one new token through the model
  ↓
reuse all earlier keys and values
  ↓
select the next token
  ↓
repeat
```

This is no longer a plan for future work. It is real cached execution on the
Apple GPU.

For the prompt token IDs:

```text
[1, 15043]
```

our cached model generates:

```text
[1789, 19053, 25470, 2323,
 21742, 13486, 13934, 26715]
```

These are exactly the same eight greedy tokens produced by:

```text
our full-recomputation model
Hugging Face's reference model
```

The important difference is the amount of model input:

```text
Lesson 36 full recomputation:
2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 44 token positions

Lesson 39 cached execution:
2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 9 token positions
```

## If you read only five things

1. A KV cache stores previous **keys and values**, not previous context
   vectors.
2. Prefill processes the whole prompt and fills every layer's cache.
3. Decode processes only one newly generated token.
4. The new token's K and V are appended **before** attention, because the
   token is allowed to attend to itself.
5. Cached decoding avoids recalculating old token states, but the new query
   must still examine all cached keys and values.

## What is stored?

For every transformer layer and every consumed token, we store:

```text
key vector
value vector
```

We do not store the token's old context vector.

### Why not store contexts?

A context vector is the result of one particular query:

```text
context = attention probabilities × values
```

Different tokens have different queries, so they produce different attention
probabilities.

For example:

```text
query for token C → one set of probabilities
query for token D → a different set of probabilities
```

Therefore the old context for `C` cannot answer the new query for `D`.

For the newest token we calculate:

```text
Q_new
  ↓ compare with every cached K
attention probabilities
  ↓ weighted sum of every cached V
context_new
```

The reusable ingredients are K and V.

That is why the structure is called a **KV cache**.

## Why old context vectors do not need to be recalculated

Llama uses causal attention.

An earlier token cannot look at a future token:

```text
context(A) depends on A
context(B) depends on A, B
context(C) depends on A, B, C
```

After token `D` arrives:

```text
context(A), context(B), context(C)
```

do not change. None of them is allowed to see `D`.

Their old hidden-state work is finished. We keep only the K and V vectors
that future tokens will need.

## Prefill

Suppose the prompt contains `P` tokens.

Prefill sends all `P` tokens through the model:

```text
input shape = [P, hidden]
```

At one attention layer:

```text
Q [H, P, D]
K [H, P, D]
V [H, P, D]
```

where:

```text
H = attention head count
D = head dimension
```

The prompt K and V tensors are appended to that layer's cache.

Attention then uses:

```text
Q [H, P, D]
K [H, P, D]

scores = Q @ Kᵀ
scores shape = [H, P, P]
```

This is square because there are:

```text
P query tokens
P key-axis positions in the score matrix
```

The causal mask still prevents each prompt query from seeing later prompt
tokens. For example, query position `0` may use only key position `0`, even
though the allocated score row has `P` columns.

## One-token decode

Assume the cache already contains `C` tokens.

The newly supplied token produces:

```text
Q_new [H, 1, D]
K_new [H, 1, D]
V_new [H, 1, D]
```

After appending the new K and V:

```text
K_all [H, C + 1, D]
V_all [H, C + 1, D]
```

Attention calculates:

```text
Q_new @ K_allᵀ

[H, 1, D] @ [H, D, C + 1]
          ↓
scores [H, 1, C + 1]
```

Then:

```text
probabilities [H, 1, C + 1]
V_all        [H, C + 1, D]
              ↓
context      [H, 1, D]
```

This is rectangular:

```text
one new query row
C + 1 key/value rows
```

The model computes only one new context vector.

## Why append happens before attention

This ordering is essential:

```text
calculate K_new and V_new
          ↓
append them to the cache
          ↓
Q_new attends over old tokens and the new token itself
```

The new token is causally allowed to see itself.

If attention ran before the append, it would see only:

```text
positions 0 ... C - 1
```

and incorrectly omit its own position `C`.

The core invariant is:

```text
query_start + query_token_count == valid_cache_length
```

For one-token decode:

```text
query_start = C
query_token_count = 1
valid_cache_length = C + 1
```

## Absolute positions and the causal mask

During prefill:

```text
query_start = 0
```

For local query row `q`:

```text
absolute query position = q
```

During decode after `C` cached tokens:

```text
query_start = C
local query row = 0
absolute query position = C
```

The cached softmax kernel permits key position `k` when:

```text
k <= absolute_query_position
```

In code, the idea is:

```cpp
const std::size_t absolute_query =
    query_start + local_query;

if (key <= absolute_query) {
    // This key is causally visible.
}
```

This one rule works for both:

```text
prefill: many query rows beginning at position 0
decode:  one query row beginning later in the sequence
```

## RoPE must also use the absolute position

RoPE changes Q and K according to token position.

For a prompt:

```text
positions = 0, 1, 2, ...
```

If the cache already contains `C` tokens, the decode token occupies position:

```text
C
```

Therefore Lesson 39 calls RoPE using:

```cpp
start_position = cache.sequence_length();
```

Using zero for every decode token would give it the wrong rotation and the
cached result would no longer match full recomputation.

## Every layer owns a separate cache

Llama does not calculate one global K and V pair.

Every transformer layer receives a different hidden state:

```text
embeddings
    ↓
layer 0 → K₀ and V₀
    ↓
layer 1 → K₁ and V₁
    ↓
layer 2 → K₂ and V₂
```

Therefore:

```text
layer 0 has its own K/V cache
layer 1 has its own K/V cache
...
```

While one model pass is running, layers advance one after another:

```text
process layer 0 → append layer 0
process layer 1 → append layer 1
...
```

Their lengths can temporarily differ inside the pass.

After a successful complete pass, every layer must have the same length.

## The per-layer execution sequence

For one input chunk, Lesson 39 performs this sequence at every layer:

```text
hidden states for only the new chunk
              ↓
           RMSNorm
              ↓
       Q, K, V projections
              ↓
         apply RoPE to Q and K
              ↓
       change layout to [H, Q, D]
              ↓
    append this chunk's K and V to cache
              ↓
 Q attends to all valid cached K and cached V
              ↓
      attention output projection
              ↓
       first residual connection
              ↓
           RMSNorm
              ↓
       SwiGLU feed-forward network
              ↓
       second residual connection
              ↓
      hidden states for next layer
```

During decode, every box operates on one new token except attention's scan
through cached K and V.

## `CachedAttention`

The small reusable attention component is:

```cpp
class CachedAttention {
public:
    RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& queries,
        const RuntimeTensor& key_storage,
        const RuntimeTensor& value_storage,
        std::size_t valid_length,
        std::size_t query_start,
        std::size_t threads_per_group = 64
    ) const;
};
```

Its inputs are:

```text
queries       [H, Q, D]
key storage   [H, capacity, D]
value storage [H, capacity, D]
```

Only this prefix of storage is valid:

```text
[0, valid_length)
```

The remaining reserved rows must not be read.

It runs three simple Metal kernels:

```text
1. calculate Q · K scores
2. causal, numerically stable softmax
3. calculate the weighted sum of V
```

This is intentionally clear rather than highly optimized.

## `CachedLlamaExecutor`

The model-facing API is:

```cpp
const CachedLlamaExecutor executor(model);
KVCache cache = executor.create_cache(capacity);

RuntimeTensor prompt_logits =
    executor.prefill(queue, prompt_ids, cache);

RuntimeTensor next_logits =
    executor.decode_one(queue, generated_token, cache);
```

The executor borrows `model`, and the generator similarly borrows its
executor. Their rvalue constructors are deleted so they cannot accidentally
retain references to temporary objects that have already been destroyed.
The borrowed model and executor must also remain alive and must not be moved
while their borrower exists.

A populated cache must stay with the executor/model that created it. Lesson
39 validates cache dimensions, but the generic `KVCache` does not yet carry a
model-identity tag. Therefore, do not pass a populated cache to another
executor merely because both models happen to have the same dimensions.

### `prefill`

`prefill` requires:

```text
a non-empty prompt
an empty cache
enough cache capacity
cache geometry matching the model
```

It returns:

```text
[prompt_tokens, vocabulary_size]
```

The generator uses the last row to choose the first generated token.

### `decode_one`

`decode_one` requires:

```text
exactly one token ID
a non-empty uniform cache
one remaining cache position
```

It returns:

```text
[1, vocabulary_size]
```

That row predicts the following token.

## The generation timeline

Suppose the prompt is:

```text
A B
```

and we want to generate:

```text
C D E
```

Execution is:

```text
prefill(A B) → choose C
decode(C)    → choose D
decode(D)    → choose E
```

Notice that `E` is selected but not sent back into the model.

It would be consumed only if we requested another prediction:

```text
decode(E) → choose F
```

Therefore generating `G` tokens uses:

```text
1 prefill
G - 1 decode calls
```

when `G > 0` and EOS does not stop generation early.

## Why final cache length is 9, not 10

Our verified example has:

```text
prompt tokens = 2
generated tokens = 8
```

The model consumes:

```text
2 prompt tokens
7 generated tokens
```

The eighth generated token is the final answer selected by the last pass. It
has not yet been used as model input.

So:

```text
final valid cache length = 2 + 7 = 9
```

We allocate capacity:

```text
2 + 8 = 10
```

because position 9 would be needed if generation continued for one more
prediction.

## Exact equivalence to full recomputation

A KV cache is an execution optimization. It must not change the mathematical
model.

The tests compare:

```text
cached prefill logits
        versus
full-forward prompt logits
```

Then after every generated token they compare:

```text
one-row cached decode logits
        versus
the final row of full-prefix forward logits
```

The tensor shapes differ:

```text
cached decode logits [1, vocabulary]
full forward logits  [complete_prefix, vocabulary]
```

but the final prediction row represents the same token position.

They agree within floating-point tolerance and choose the same greedy token
at all eight steps.

## What work the cache removes

Without caching, every generation step recalculates old-token work:

```text
old embeddings
old normalization
old Q/K/V projections
old attention outputs
old feed-forward layers
```

With caching, decode runs those operations only for the newest token.

For the eight-token example:

```text
old path:    44 input token positions
cached path:  9 input token positions
```

That is the central saving implemented by this lesson.

## What the cache does not remove

The newest query still needs to decide how much attention to give every
earlier token:

```text
Q_new @ K_cacheᵀ
```

and then collect information from:

```text
V_cache
```

If there are `C` earlier tokens, decode attention still scans roughly `C + 1`
keys and values.

Therefore KV caching changes:

```text
recalculate the whole transformer for all C tokens
```

into:

```text
calculate one new token
+ scan C cached attention positions
```

It does not make attention constant-time.

## Actual statistics

`CachedGenerationStats` records work that really happened:

```cpp
prefill_pass_count
decode_pass_count
model_pass_count
model_input_token_positions
final_cache_length
cache_capacity
```

For prompt length 2 and eight generated tokens:

```text
prefill passes               1
decode passes                7
total model passes           8
model input token positions  9
final cache length           9
cache capacity              10
```

Unlike Lesson 37, these are not estimates for a future implementation.

## Error checks

The runtime rejects important invalid states:

```text
empty prompt
out-of-range token ID
prefill into a non-empty cache
decode before prefill
prompt larger than cache capacity
decode beyond cache capacity
cache geometry that does not match the model
non-uniform per-layer lengths
invalid query/cache shapes
query positions inconsistent with valid cache length
generation beyond maximum model position
invalid sampling temperature or top-k
```

These checks matter because an incorrect cache position can silently produce
plausible-looking but wrong tokens.

## What is deliberately still simple

This is a correctness-first inference engine:

```text
batch size one
f32 tensors
ordinary multi-head attention only
one contiguous cache allocation per K or V tensor
three separate cached-attention kernels
many temporary tensors
many synchronous waits
naive linear and attention kernels
no paged cache
no continuous batching
no request scheduler
```

So Lesson 39 proves the architecture and numerical correctness. It is not yet
a production-speed vLLM replacement.

## Files

### `cached_llama.hpp` and `cached_llama.cpp`

Contain:

```text
CachedAttention
CachedLlamaExecutor
prefill
decode_one
layer-by-layer cache integration
```

### `cached_generation.hpp` and `cached_generation.cpp`

Contain:

```text
the append-and-repeat generation loop
greedy selection
temperature and top-k sampling
EOS stopping
actual cached-execution statistics
```

### `demo.cpp`

Loads the real tiny Hugging Face model and generates token IDs using the
cached path.

### `tests.cpp`

Checks:

```text
synthetic prefill attention against a CPU reference
synthetic one-token decode against a CPU reference
prefill logits against full model forward
every cached decode row against full-prefix recomputation
the exact eight greedy token IDs
1 prefill + 7 decode calls
9 cached input positions instead of 44
final cache length 9
zero-token generation
EOS stopping
capacity and state errors
seeded sampling parity with Lesson 36
```

## Recommended reading order

You do not need to read every Metal line.

Read these parts in order:

1. The comments and public APIs in `cached_llama.hpp`.
2. `CachedLlamaExecutor::prefill()`.
3. `CachedLlamaExecutor::decode_one()`.
4. The per-layer loop in `CachedLlamaExecutor::run_chunk()`.
5. The three dispatches in `CachedAttention::run()`.
6. `CachedAutoregressiveGenerator::generate()`.
7. The synthetic CPU-reference tests if you want to verify the mathematics.

The most important line of thought in `run_chunk()` is:

```text
project Q/K/V
apply RoPE using query_start
append K/V
attend over the complete valid cache
continue the transformer block
```

## Build the demo

Run from the `infer-lab` repository directory:

```bash
xcrun clang++ \
  -std=c++20 -O2 \
  -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wsign-conversion -fobjc-arc \
  phase0/lesson21-types-shapes-broadcasting/miniir.cpp \
  phase0/lesson25-tensor-storage-layout/runtime_tensor.cpp \
  phase0/lesson26-tensor-views/tensor_views.cpp \
  phase0/lesson28-contiguous-buffer-reuse/contiguous.cpp \
  phase0/lesson31-linear-operator/linear.cpp \
  phase0/lesson32-batched-matmul/batched_matmul.cpp \
  phase0/lesson33-transformer-ops/transformer_ops.cpp \
  phase0/lesson34-attention-transformer-block/transformer_block.cpp \
  phase0/lesson35-huggingface-llama/hf_loader.cpp \
  phase0/lesson35-huggingface-llama/hf_llama.cpp \
  phase0/lesson36-autoregressive-generation/generation.cpp \
  phase0/lesson38-kv-cache-layout/kv_cache.cpp \
  phase0/lesson39-cached-decoding/cached_llama.cpp \
  phase0/lesson39-cached-decoding/cached_generation.cpp \
  phase0/lesson14-metal-minigpu/minigpu.mm \
  phase0/lesson39-cached-decoding/demo.cpp \
  -framework Foundation -framework Metal \
  -o phase0/work/lesson39_cached_generate
```

## Run the real model

```bash
phase0/work/lesson39_cached_generate \
  models/hf-tiny-random-llama \
  greedy \
  8 \
  phase0/work/lesson39_ids.txt \
  1 15043
```

The generated IDs should be:

```text
[1789, 19053, 25470, 2323,
 21742, 13486, 13934, 26715]
```

The output file contains the prompt and generated IDs together.

## Run through the tokenizer bridge

Lesson 36's Python bridge accepts the same command-line interface:

```bash
python3 phase0/lesson36-autoregressive-generation/run_prompt.py \
  models/hf-tiny-random-llama \
  phase0/work/lesson39_cached_generate \
  "Hello" \
  --max-new-tokens 8
```

Python performs tokenization and text decoding. The C++/Metal program still
performs model inference and token generation.

## Build and run the tests

Use the same source list as the demo, but replace:

```text
phase0/lesson39-cached-decoding/demo.cpp
```

with:

```text
phase0/lesson39-cached-decoding/tests.cpp
```

and name the output:

```text
phase0/work/lesson39_tests
```

Then run:

```bash
phase0/work/lesson39_tests \
  models/hf-tiny-random-llama
```

Expected result:

```text
Lesson 39 cached-decoding tests passed.
```

## Main takeaways

```text
KV cache = old keys and values, not old contexts

prefill = many new queries against the prompt cache

decode = one new query against the growing cache

append before attention = include the current token itself

absolute position = correct RoPE and causal visibility

per-layer cache = every transformer layer has different K/V

same logits = caching changes execution, not model mathematics

9 versus 44 = old-token transformer work is no longer repeated
```

## Failure safety

One model pass appends to the KV cache one layer at a time:

```text
append layer 0
append layer 1
...
append final layer
produce logits
```

If a later operation throws, Lesson 39 restores every layer to the logical
length it had before the pass. Bytes already copied beyond that length are
left in the preallocated buffers, but they are invalid and therefore cannot
be read. A retry will overwrite them.

This makes each `prefill()` or `decode_one()` call transactional from the
caller's point of view:

```text
success → all layers advance together
failure → all layers return to the old length
```

Lesson 39 deliberately does not place a model-identity tag in `KVCache`.
The caller must keep each populated cache with its creating executor. Lesson
40 will make this ownership relationship more explicit through per-request
engine state.

## Next lesson

Lesson 39 handles one request with one contiguous KV cache.

Lesson 40 begins the Mini-vLLM layer:

```text
request arrives
    ↓
sequence object tracks its state
    ↓
prompt becomes ready for prefill
    ↓
generated token becomes ready for decode
    ↓
EOS, cancellation, failure, or length limit finishes it
```

That request/sequence lifecycle will later let a scheduler manage many
independent generations.

The complete paused-development plan is in
[the future roadmap](../../FUTURE_ROADMAP.md).
