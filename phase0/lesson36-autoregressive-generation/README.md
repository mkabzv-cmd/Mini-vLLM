# Lesson 36 — Autoregressive text generation

## Result

Lesson 35 could run one complete Llama forward pass and return logits.

Lesson 36 turns those logits into a sequence of new tokens:

```text
prompt token IDs
       ↓
run the complete model
       ↓
take the last token's logits
       ↓
choose one next token
       ↓
append it to the sequence
       ↓
repeat
```

For the prompt:

```text
Hello
```

the Hugging Face tokenizer produces:

```text
[1, 15043]
```

Our C++/Metal runtime greedily generates:

```text
[1789, 19053, 25470, 2323, 21742, 13486, 13934, 26715]
```

The complete token sequence is:

```text
[1, 15043, 1789, 19053, 25470, 2323, 21742, 13486, 13934, 26715]
```

The tiny checkpoint contains random weights, so the decoded text is
nonsense. The important result is that the generation loop works and its
greedy IDs exactly match the official Hugging Face model.

## The most important idea

An LLM does not produce an entire answer in one operation.

It predicts **one next token at a time**.

Suppose the current sequence is:

```text
[1, 15043]
```

The model returns logits with shape:

```text
[2, 32000]
```

Why?

```text
2       = number of input token positions
32000   = number of possible vocabulary tokens
```

Every row contains vocabulary scores for one position. For generation, we
only need the final row:

```text
last row = scores for the token that should follow the entire prompt
```

The largest score belongs to token `1789`, so we append it:

```text
[1, 15043] + [1789]
        ↓
[1, 15043, 1789]
```

Then we run the model again to predict the token after that:

```text
[1, 15043, 1789]
        ↓ model
19053
        ↓ append
[1, 15043, 1789, 19053]
```

This repeated feedback loop is called **autoregressive generation**:

```text
the model's previous output becomes part of its next input
```

## The central C++ loop

The important part of `AutoregressiveGenerator::generate()` is conceptually:

```cpp
tokens = prompt;

for (each requested new token) {
    logits = model.forward(queue, tokens);
    last_logits = final_row(logits);
    next_token = choose(last_logits);
    tokens.push_back(next_token);

    if (next_token == eos_token) {
        break;
    }
}
```

There are four separate responsibilities:

```text
HuggingFaceLlama       calculates logits
generation policy      chooses a token from logits
generation loop        appends the token and repeats
tokenizer              converts between text and token IDs
```

Keeping them separate is useful. The same model can use greedy decoding,
sampling, beam search, or another policy without changing its transformer
implementation.

## Greedy decoding

Greedy decoding chooses the token with the largest logit:

```text
token 8      logit 0.2
token 27     logit 1.4   ← largest
token 91     logit 0.7

choose token 27
```

In code, this is an `argmax`:

```cpp
next_token_id = index_of_largest_logit;
```

We do not need softmax for greedy decoding. Softmax changes scores into
probabilities, but it does not change which finite score is largest.

If two logits are exactly equal, our implementation chooses the lower token
ID. That gives deterministic behavior.

## Sampling

Always choosing the largest logit can make a useful language model repetitive
or overly predictable.

Sampling treats the logits as a probability distribution and makes a random
choice:

```text
logits
   ↓ temperature
scaled logits
   ↓ optional top-k filtering
selected candidates
   ↓ stable softmax
probabilities
   ↓ random draw
one token
```

### Temperature

Temperature controls how strongly sampling prefers the largest logits:

```text
temperature below 1   sharper, more predictable
temperature = 1       unchanged distribution
temperature above 1   flatter, more random
```

The scaled score is:

```text
scaled_logit = logit / temperature
```

A temperature of zero is invalid for sampling.

### Top-k

Top-k keeps only the `k` highest-scoring tokens before sampling:

```text
top_k = 20
        ↓
discard every candidate except the best 20
        ↓
sample among those 20
```

In our API:

```text
top_k = 0   keep the complete vocabulary
top_k > 0   keep only that many candidates
```

### Seed

Sampling uses a pseudo-random-number generator.

The same:

```text
prompt + model + settings + seed
```

produces the same token sequence in our implementation. This is important for
repeatable tests.

## EOS: when generation stops

EOS means **end of sequence**.

After selecting a new token, the loop asks:

```cpp
if (next_token_id == eos_token_id) {
    stop;
}
```

It checks the newly generated token, not old tokens already present in the
prompt.

The API normally uses the EOS ID from `config.json`. It also permits an
override so tests or callers can choose a different stopping token.

The bundled random checkpoint is unusual:

```text
config.json says EOS ID             1
tokenizer_config.json says EOS ID   2
```

This is test-model metadata, not a rule of Llama. Our model runtime follows
`config.json` by default. One test overrides EOS with `1789` and proves that
generation stops immediately after producing that token.

## Why this first version is deliberately inefficient

Lesson 36 reruns the complete sequence before every new token.

For prompt length `P` and `G` generated tokens, the transformer receives:

```text
P, P + 1, P + 2, ... P + G - 1
```

For our `"Hello"` example:

| Prediction | Sequence length recomputed |
|---:|---:|
| token 1 | 2 |
| token 2 | 3 |
| token 3 | 4 |
| token 4 | 5 |
| token 5 | 6 |
| token 6 | 7 |
| token 7 | 8 |
| token 8 | 9 |

Total token positions processed:

```text
2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 44
```

But most of those positions were already calculated during an earlier pass.

A cache-aware implementation can process:

```text
first prediction:   the 2-token prompt
later predictions:  one newly appended token each

2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 9
```

The exact work inside attention is more complicated because each new query
still attends to earlier keys and values. But the transformer no longer has
to recreate all earlier hidden states, queries, keys, and values.

This repeated work is the reason we will build a **KV cache**.

The runtime reports both numbers:

```text
full-recomputation token positions: 44
cache-aware token positions:         9
```

## What the statistics mean

`GenerationStats` contains:

```cpp
prompt_token_count
generated_token_count
forward_pass_count
model_input_token_count
elapsed_milliseconds
```

`model_input_token_count` is especially important.

It is the sum of every sequence length sent through the full model:

```text
prompt length 3, generate 3 tokens

model inputs:
3, then 4, then 5

model_input_token_count = 3 + 4 + 5 = 12
```

This gives us a simple correctness measure for the optimization in later
lessons.

The timer starts immediately before the first generation step and stops after
the final token is selected. It includes model forwards, the synchronization
and host readback performed by `to_host()`, and CPU token selection. It does
not include model loading, prompt tokenization, or final text decoding.

Treat this lesson's time as a rough observation, not a benchmark. The stable
measurement for later comparisons is the amount of repeated token work.

## The tokenizer bridge

Our C++ runtime accepts token IDs. It does not yet implement a Llama tokenizer.

`run_prompt.py` provides a small bridge:

```text
human prompt
     ↓ Hugging Face tokenizer
prompt token IDs
     ↓ our C++/Metal generator
complete output IDs
     ↓ Hugging Face tokenizer
decoded text
```

Python performs tokenization and decoding only. The model forward passes and
token selection still happen in our C++/Metal runtime.

`reference_generation.py` is different: it runs the official Hugging Face
model with full-prefix recomputation. We use it only as a correctness oracle.

## Public API

The main configuration is:

```cpp
GenerationConfig config;
config.max_new_tokens = 8;
config.do_sample = false;  // greedy
```

For deterministic sampling:

```cpp
GenerationConfig config;
config.max_new_tokens = 8;
config.do_sample = true;
config.temperature = 0.8f;
config.top_k = 20;
config.seed = 1234;
```

Then:

```cpp
AutoregressiveGenerator generator(model);

GenerationResult result =
    generator.generate(queue, prompt_token_ids, config);
```

The result contains:

```text
token_ids             prompt plus generated tokens
generated_token_ids   only the newly generated tokens
stopped_on_eos        whether EOS ended the loop
stats                 work and timing information
```

## Build

Run this from the workspace root:

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
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o outputs/infer-lab/phase0/work/lesson36_generate
```

## Run a human-readable prompt

```bash
python3 \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/run_prompt.py \
  outputs/infer-lab/models/hf-tiny-random-llama \
  outputs/infer-lab/phase0/work/lesson36_generate \
  "Hello" \
  --max-new-tokens 8
```

Use deterministic sampling instead:

```bash
python3 \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/run_prompt.py \
  outputs/infer-lab/models/hf-tiny-random-llama \
  outputs/infer-lab/phase0/work/lesson36_generate \
  "Hello" \
  --max-new-tokens 8 \
  --mode sample
```

## Run the official reference

```bash
python3 \
  outputs/infer-lab/phase0/lesson36-autoregressive-generation/reference_generation.py \
  outputs/infer-lab/models/hf-tiny-random-llama \
  "Hello" \
  --max-new-tokens 8
```

## Files worth reading

Read only these parts first:

1. `generation.hpp` — the small public API and statistics.
2. `AutoregressiveGenerator::generate()` — the append-and-repeat loop.
3. `select_greedy()` — choose the largest logit.
4. `select_sampled()` — temperature, top-k, softmax, and random choice.
5. `run_prompt.py` — only if you want to see the tokenizer bridge.

You do not need to study all validation code now.

## What comes next

Lesson 36 proves that generation works, but also exposes its largest waste:

```text
the same prompt tokens are recomputed again and again
```

The next steps are:

```text
Lesson 37   prefill versus decode
Lesson 38   KV-cache theory and memory layout
Lesson 39   cached one-token decoding
```

Those lessons begin the transition from a correct model runner to a real
inference engine.
