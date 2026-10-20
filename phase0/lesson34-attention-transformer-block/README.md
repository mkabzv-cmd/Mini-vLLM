# Lesson 34 — Attention and a complete transformer block

## What this lesson builds

This is the first lesson where the earlier pieces become one recognizable
LLM component.

```text
hidden states
    ↓
RMSNorm
    ↓
causal multi-head self-attention
    ↓
residual addition
    ↓
RMSNorm
    ↓
SwiGLU feed-forward network
    ↓
residual addition
    ↓
next hidden states
```

This is a miniature Llama-style decoder block.

It is correctness-first, not fast:

```text
batch size                 = 1
ordinary multi-head attention
separate GPU kernels
synchronous helper calls
no KV cache yet
```

## First: what is attention?

Every token is represented by a vector.

Suppose:

```text
tokens = 3
hidden size = 4
```

Then the hidden-state tensor has shape:

```text
[tokens, hidden] = [3,4]
```

Conceptually:

```text
token 0 → [h00, h01, h02, h03]
token 1 → [h10, h11, h12, h13]
token 2 → [h20, h21, h22, h23]
```

Attention lets every token gather useful information from other allowed
tokens.

## Query, key, and value

The same hidden state is projected three different ways:

```text
Q = hidden @ query_weightᵀ
K = hidden @ key_weightᵀ
V = hidden @ value_weightᵀ
```

An intuitive interpretation is:

```text
Query → what is this token looking for?
Key   → what kind of information does this token contain?
Value → what information should be returned if selected?
```

The query and key decide the importance.

The value carries the content.

These are learned linear projections:

```cpp
RuntimeTensor query = linear.run(
    queue,
    hidden,
    query_weight
);
```

The Hugging Face model will eventually provide these weight tensors.

`causal_self_attention()` expects this input to have already passed through
RMSNorm. The full `run()` method performs that normalization for you. The
demo also calls attention directly with raw values only to test that
subroutine in isolation.

## Splitting into heads

Instead of one large comparison, attention divides the hidden dimension into
smaller heads.

For:

```text
hidden = 4
heads = 2
```

each head has:

```text
head dimension = hidden / heads = 2
```

The projection initially has shape:

```text
[tokens, hidden] = [3,4]
```

We reinterpret the same compact storage as:

```text
[tokens, heads, head_dimension] = [3,2,2]
```

Then transpose and materialize it as:

```text
[heads, tokens, head_dimension] = [2,3,2]
```

Now each head is one batch for Lesson 32's batched matmul.

## Why multiple heads?

Different heads can learn different relationships.

For example, one head might become useful for:

```text
nearby syntax
```

while another might become useful for:

```text
longer-range references
```

The runtime does not assign those meanings. Training determines what the
weight matrices make each head learn.

## RoPE is applied to Q and K

Lesson 33's rotary position embedding changes queries and keys according to
their token position:

```text
Q → RoPE(Q)
K → RoPE(K)
```

Values are not rotated.

RoPE lets the query-key dot product carry relative-position information.

Here, `start_position` changes only those RoPE positions. There is no KV
cache yet, so a supplied chunk cannot attend to tokens that came before the
chunk. We will add that history when we implement cached decoding.

## Computing attention scores

For each head:

```text
scores = Q × transpose(K)
```

Shapes:

```text
Q                 [heads, query_tokens, head_dimension]
K                 [heads, key_tokens,   head_dimension]
transpose(K)      [heads, head_dimension, key_tokens]
scores            [heads, query_tokens, key_tokens]
```

For self-attention:

```text
query_tokens = key_tokens = tokens
```

Our Lesson 32 call is:

```cpp
RuntimeTensor scores = batched_matmul.run(
    queue,
    query_heads,
    key_heads,
    true // transpose the right operand logically
);
```

One score is a dot product:

```text
score(query_token, key_token)
    = Σ Q[query_token, d] × K[key_token, d]
```

A larger score means the query and key match more strongly.

## Why divide by the square root?

Raw dot products tend to grow as the head dimension grows.

Attention scales them using:

```text
scaled score =
    score / sqrt(head_dimension)
```

Without scaling, softmax can become extremely sharp, which makes numerical
behavior and training more difficult.

## The causal mask

During language generation, a token must not see future tokens.

For three tokens, allowed attention looks like:

```text
             key 0   key 1   key 2
query 0        yes      no      no
query 1        yes     yes      no
query 2        yes     yes     yes
```

Or as a mask:

```text
[
  [allowed, blocked, blocked],
  [allowed, allowed, blocked],
  [allowed, allowed, allowed]
]
```

The Metal kernel implements:

```metal
if (key <= query) {
    // Include this score.
}
```

Blocked probabilities are written as zero.

The demo checks causality by changing every feature of tokens `1` and `2`
and checking that token `0` remains the same within floating-point
tolerance.

## Softmax turns scores into probabilities

For every query row:

```text
scores → causal mask → stable softmax → probabilities
```

The probabilities:

```text
are non-negative
sum to approximately 1 across allowed keys
are zero for future keys
```

The kernel uses the numerically stable formula:

```text
exp(score - maximum_score)
```

It performs two threadgroup reductions:

```text
first reduction  → row maximum
second reduction → exponential sum
```

All 64 lanes reach every barrier, including lanes that have no key to
process.

## Using the values

The probabilities decide how much of every value vector to collect:

```text
context = probabilities × V
```

Shapes:

```text
probabilities [heads, tokens, tokens]
V             [heads, tokens, head_dimension]
context       [heads, tokens, head_dimension]
```

For one query token:

```text
context =
    probability_0 × value_0
  + probability_1 × value_1
  + ...
```

Earlier tokens with higher probability contribute more.

## Joining the heads

The context is changed from:

```text
[heads, tokens, head_dimension]
```

back into:

```text
[tokens, heads, head_dimension]
```

and then reinterpreted as:

```text
[tokens, hidden]
```

An output projection mixes information across the heads:

```text
attention output =
    joined context @ attention_output_weightᵀ
```

## The first residual connection

The attention output is added to the block input:

```text
after_attention =
    original_hidden + attention_output
```

This is called a residual or skip connection.

It gives information a direct path through the network instead of forcing
every value to be completely reconstructed by attention.

## The feed-forward network

After another RMSNorm, a Llama-style block uses a gated MLP:

```text
gate = normalized @ gate_weightᵀ
up   = normalized @ up_weightᵀ

hidden_gate = SiLU(gate) × up

down = hidden_gate @ down_weightᵀ
```

This is commonly called SwiGLU:

```text
SwiGLU(x) = SiLU(gate(x)) × up(x)
```

The intermediate dimension is normally larger than the hidden dimension.

Our tiny demonstration uses:

```text
hidden       = 4
intermediate = 6
```

Real models use much larger dimensions.

## The second residual connection

The MLP result is added back:

```text
block_output =
    after_attention + down
```

The complete block is therefore:

```text
x
│
├─────────────── residual ───────────────┐
│                                        │
└→ RMSNorm → attention ──────────────────+→ x1
                                         │
                                         ├── residual
                                         │
                      RMSNorm → SwiGLU ──+→ output
```

## Important code sequence

The main architecture appears directly in `run()`:

```cpp
attention_input = rms_norm(hidden);
attention_output = causal_self_attention(
    attention_input
);
after_attention = hidden + attention_output;

feed_forward_input = rms_norm(after_attention);
gate = linear(feed_forward_input, gate_weight);
up = linear(feed_forward_input, up_weight);
gated = silu(gate) * up;
down = linear(gated, down_weight);

output = after_attention + down;
```

This sequence is more important than the helper and validation code.

## How the earlier lessons are used

```text
Lesson 28 → materialize transposed head layouts
Lesson 31 → Q, K, V, output, gate, up, down projections
Lesson 32 → QKᵀ and probability-times-V batched matmuls
Lesson 33 → RMSNorm, RoPE, and SiLU
Lesson 34 → causal softmax, residuals, orchestration
```

This is why we built the stack hierarchically.

## The block's weight object

```cpp
struct TransformerBlockWeights {
    RuntimeTensor attention_norm_weight;
    RuntimeTensor query_weight;
    RuntimeTensor key_weight;
    RuntimeTensor value_weight;
    RuntimeTensor attention_output_weight;

    RuntimeTensor ffn_norm_weight;
    RuntimeTensor gate_weight;
    RuntimeTensor up_weight;
    RuntimeTensor down_weight;
};
```

These names correspond approximately to Llama Hugging Face tensors such as:

```text
input_layernorm.weight
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
post_attention_layernorm.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

The Hugging Face loader will fill these RuntimeTensors.

## What comes from the model configuration?

The future loader will read values such as:

```text
hidden_size
num_attention_heads
intermediate_size
rms_norm_eps
rope_theta
number of layers
```

The configuration describes the architecture.

The weight files contain the learned numbers.

Our code supplies the execution machinery.

## Why this implementation has many dispatches

Every helper currently submits and waits:

```text
linear dispatch
RoPE dispatch
materialization dispatch
batched matmul dispatch
softmax dispatch
...
```

This is intentionally easy to verify, but not how a fast production engine
should schedule the block.

Later improvements include:

```text
asynchronous dependency chaining
kernel fusion
buffer planning and reuse
fused QKV projection
fused attention
KV cache
prefill/decode-specific kernels
```

## Why the block materializes head layouts

Q, K, and V start in token-major form:

```text
[tokens, heads, head_dimension]
```

Lesson 32's batched matmul expects each head as a compact batch:

```text
[heads, tokens, head_dimension]
```

Transpose creates the correct logical view but not contiguous physical
storage. Lesson 28 materializes it.

This costs extra copies. A later attention kernel can understand the original
strides directly and avoid some materializations.

## What the demo checks

The demo contains an independent CPU implementation and compares:

```text
causal attention output
complete transformer-block output
```

It also checks:

```text
future tokens cannot affect token zero
nonzero RoPE start positions pass through the complete block
output shape is [tokens, hidden]
output uses compact storage
heads must divide hidden size
RoPE head dimension must be even
feed-forward weight shapes must agree
```

## What to read

Start with:

1. The attention overview in this README.
2. `MiniTransformerBlock::causal_self_attention()`.
3. `MiniTransformerBlock::run()`.
4. The causal-softmax MSL kernel.
5. The final CPU/GPU comparisons in `demo.cpp`.

Skim:

```text
large test weight values
CPU reference loops
shape-validation helpers
printing helpers
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
  outputs/infer-lab/phase0/lesson31-linear-operator/linear.cpp \
  outputs/infer-lab/phase0/lesson32-batched-matmul/batched_matmul.cpp \
  outputs/infer-lab/phase0/lesson33-transformer-ops/transformer_ops.cpp \
  outputs/infer-lab/phase0/lesson34-attention-transformer-block/transformer_block.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson34-attention-transformer-block/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o work/lesson34_attention_transformer_block

./work/lesson34_attention_transformer_block
```

## Main takeaways

1. Q and K calculate relevance; V carries the collected information.
2. Heads split the hidden dimension into several attention spaces.
3. RoPE adds position information to Q and K.
4. Scaling, the causal mask, and softmax turn scores into valid attention
   probabilities.
5. Probability-weighted values produce the attention context.
6. Residual connections preserve a direct information path.
7. SwiGLU is the Llama-style gated feed-forward network.
8. A transformer block is orchestration of familiar tensor operations, not
   one mysterious instruction.

## Deliberately deferred

```text
batch sizes greater than one
grouped-query attention
KV cache
padding masks
dropout
f16/bfloat16
fused kernels
asynchronous whole-block execution
automatic temporary-buffer reuse
multiple stacked layers
token embedding and final language-model head
```

## After these lessons

The next phase is exactly what you requested:

```text
load Hugging Face configuration
load safetensors weights
map model tensor names into RuntimeTensors
run token embedding
execute one or more transformer blocks
apply final normalization and LM head
produce logits
```

After that we will add:

```text
prefill
one-token decode
KV cache
continuous batching
paged KV cache
Mini-vLLM scheduling
```
