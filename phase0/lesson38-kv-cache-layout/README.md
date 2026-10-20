# Lesson 38 — KV-cache storage and append

## Result

Lesson 37 described the execution we want:

```text
prefill(prompt)
    ↓
save the prompt's keys and values
    ↓
decode one new token
    ↓
reuse all earlier keys and values
```

Lesson 38 builds the storage in the middle of that picture.

We now have a real `KVCache` that:

```text
allocates Metal K and V buffers accessible by the Apple GPU
keeps separate storage for every transformer layer
tracks how many token rows are valid in each layer
appends a whole prefill chunk
appends one decode token
checks capacity and tensor shapes
reports its memory use
resets without reallocating
```

The append operation is a real Metal kernel.

Our current MiniGPU allocator uses Metal's shared storage mode. On Apple
Silicon, CPU and GPU access the same unified physical memory; this is not a
separate NVIDIA-style VRAM allocation. Calling these "GPU buffers" means
they are Metal resources that GPU kernels can directly access. The CPU must
still synchronize with GPU work before safely reading results, which is why
the synchronous `append()` waits for its event.

However, this lesson has one deliberate boundary:

```text
Lesson 38 stores K and V
Lesson 39 makes attention read K and V from the cache
```

The Hugging Face model and Lesson 36 generator still use full-prefix
recomputation in this lesson. We are building and testing the cache as an
independent lower-level component before connecting it to attention.

## The shortest mental model

Imagine one transformer layer has processed three tokens:

```text
token positions:     0       1       2

key cache:          K₀      K₁      K₂
value cache:        V₀      V₁      V₂
```

The next token produces:

```text
Q₃, K₃, V₃
```

We append:

```text
key cache:          K₀      K₁      K₂      K₃
value cache:        V₀      V₁      V₂      V₃
```

Then the new query can attend to all four positions:

```text
scores₃ = Q₃ @ [K₀ K₁ K₂ K₃]ᵀ

context₃ =
    probability₀ × V₀
  + probability₁ × V₁
  + probability₂ × V₂
  + probability₃ × V₃
```

This is the entire purpose of the cache: retain the earlier `K` and `V`
vectors so that we do not recreate them during every decode step.

## What exactly does the cache store?

It stores:

```text
the key vectors for every processed token
the value vectors for every processed token
```

It does **not** normally store:

```text
queries
attention scores
softmax probabilities
context vectors
logits
```

### Why not cache queries?

A query is needed when its own token is being processed:

```text
Q_new asks:
"Which earlier keys are relevant to me?"
```

Future tokens do not need `Q_new`. They create their own queries and compare
those queries with the saved keys.

Therefore:

```text
Q is used now
K is useful to future queries
V is useful to future weighted sums
```

### Why not cache context vectors?

The context vector belongs to one particular query:

```text
context_for_token_3 =
    probabilities_from_Q3 × all_visible_values
```

The next token has a different query:

```text
context_for_token_4 =
    probabilities_from_Q4 × all_visible_values
```

Because `Q3` and `Q4` are different, their probabilities and context vectors
are generally different.

Earlier context vectors have already been used to finish their own tokens'
layer computations. Causal attention also means a later token cannot change
an earlier token's context.

What the new token needs from the past is therefore:

```text
past K vectors → calculate new attention scores
past V vectors → calculate new context
```

That is why this is called a **KV cache**, not a query cache or context
cache.

## Why every transformer layer needs its own cache

A transformer has several layers:

```text
layer 0
   ↓
layer 1
   ↓
layer 2
   ↓
...
```

Each layer receives a different hidden state and owns different projection
weights. Consequently, layer 0 and layer 1 do not produce the same keys and
values for a token:

```text
layer 0: K⁰ₜ, V⁰ₜ
layer 1: K¹ₜ, V¹ₜ
layer 2: K²ₜ, V²ₜ
```

One shared pair of K/V tensors for the entire model would therefore be
incorrect.

Our cache is conceptually:

```text
KVCache
├── layer 0
│   ├── key storage
│   └── value storage
├── layer 1
│   ├── key storage
│   └── value storage
└── ...
```

## The four important dimensions

We use:

```text
L   = number of transformer layers
Hkv = number of key/value heads
C   = maximum token capacity
D   = head dimension
```

The configuration is:

```cpp
struct KVCacheConfig {
    std::size_t layer_count;
    std::size_t key_value_head_count;
    std::size_t capacity;
    std::size_t head_dimension;
};
```

For every layer, we allocate two tensors:

```text
K storage [Hkv, C, D]
V storage [Hkv, C, D]
```

Our miniature runtime has batch size one, so there is no batch dimension.
A production cache often has a shape related to:

```text
[batch, Hkv, C, D]
```

Production engines may use a paged physical layout instead, but these logical
dimensions still exist.

## What does one cached element mean?

Consider:

```text
K[layer, head, token, component]
```

It selects:

```text
one transformer layer
one key/value head
one token position
one number inside that head's vector
```

The layer is represented by a separate tensor in our implementation. Inside
one layer's tensor, the logical coordinates are:

```text
[head, token, component]
```

## Turning three coordinates into one buffer index

The Metal allocation is ultimately a flat array of floats.

For one layer, we flatten:

```text
[Hkv, C, D]
```

with:

```text
index =
    ((head * C) + token) * D + component
```

The code calls the same calculation:

```cpp
cache.storage_index(head, token, component);
```

For example, suppose:

```text
C = 4
D = 2

head      = 1
token     = 2
component = 0
```

Then:

```text
index = ((1 × 4) + 2) × 2 + 0
      = 12
```

The capacity appears in the formula because each head reserves room for all
`C` token rows:

```text
head 0: tokens 0 ... C-1
head 1: tokens 0 ... C-1
...
```

## Capacity and valid length are different

This distinction is central.

### Capacity

Capacity is how many token rows have been allocated:

```text
capacity = 8
```

The physical GPU buffers already reserve eight positions per KV head.

### Valid length

Valid length is how many positions currently contain usable K/V data:

```text
valid length = 3
```

Only token positions:

```text
0, 1, 2
```

are logically part of the sequence.

The remaining positions are reserved but invalid:

```text
capacity:      [0] [1] [2] [3] [4] [5] [6] [7]
valid rows:     ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗
```

If we append two tokens, they enter positions 3 and 4:

```text
capacity:      [0] [1] [2] [3] [4] [5] [6] [7]
valid rows:     ✓   ✓   ✓   ✓   ✓   ✗   ✗   ✗

new valid length = 5
```

The cache rejects an append when:

```text
old valid length + new token count > capacity
```

## Why preallocate the capacity?

Without preallocation, every generated token could require:

```text
allocate a larger GPU buffer
copy all old keys
copy all old values
append the new key and value
release the old buffer
```

Doing that on every decode step would add allocation and copying overhead to
the latency-sensitive token loop.

Instead, we allocate once:

```text
[Hkv, capacity, D]
```

and append into the next free token rows:

```text
destination_start = current valid length
```

This is simple and fast for one sequence.

It can waste memory when the requested capacity is much larger than the
actual sequence. Later, paged attention will solve that different problem by
allocating cache blocks on demand.

## Prefill appends a chunk

Suppose the prompt has four tokens:

```text
A B C D
```

One layer calculates compact tensors:

```text
new K [Hkv, 4, D]
new V [Hkv, 4, D]
```

The layer begins with:

```text
valid length = 0
```

One append copies the complete prompt chunk into positions:

```text
0, 1, 2, 3
```

After the append:

```text
valid length = 4
```

The API does not require prefill to contain the entire prompt in one call.
Chunked prefill can append several chunks:

```text
append 3 tokens → length 3
append 2 tokens → length 5
append 4 tokens → length 9
```

Every append begins at the current valid length.

## Decode appends one token

After prefill, suppose:

```text
valid length = 4
```

The newly generated token produces:

```text
new K [Hkv, 1, D]
new V [Hkv, 1, D]
```

Appending them writes token position 4:

```text
before: K₀ K₁ K₂ K₃
append:             K₄
after:  K₀ K₁ K₂ K₃ K₄
```

and similarly for `V`.

The new valid length becomes:

```text
5
```

The same append mechanism therefore supports both phases:

```text
prefill → new_tokens may be many
decode  → new_tokens is normally one
```

## The real Metal append kernel

The source chunk is compact:

```text
[Hkv, new_tokens, D]
```

The destination is preallocated:

```text
[Hkv, capacity, D]
```

Those tensors use different token strides. We cannot simply treat the
destination as if its second dimension were `new_tokens`.

The kernel launches:

```text
Hkv × new_tokens × D
```

logical work items.

One GPU lane identifies:

```text
head
token inside the new chunk
component inside the head vector
```

It computes:

```text
source index      = global_id

destination index =
    ((head * capacity)
      + destination_start
      + token)
    * head_dimension
    + component
```

Then that lane copies both corresponding numbers:

```metal
key_cache[destination_index] =
    new_keys[global_id];

value_cache[destination_index] =
    new_values[global_id];
```

Different lanes write different destination indexes, so the copy does not
need atomics or threadgroup barriers.

### Why does `append()` wait?

`append()` submits the Metal dispatch and waits for its event:

```cpp
minigpu::Event finished =
    queue.submit(std::move(dispatch), launch);

finished.wait();
```

Only after the GPU command succeeds does the CPU update:

```text
valid length = old length + new token count
```

This gives a simple guarantee:

```text
if append() returns successfully,
the copied bytes exist and the new length is valid
```

If another GPU kernel produced the K/V chunk earlier on the same queue, FIFO
ordering makes the append wait behind that producer automatically. If a
different queue produced the chunk, the caller must first synchronize those
queues; `append()` cannot infer cross-queue dependencies by itself.

An asynchronous production implementation can avoid this host wait, but it
must connect cache metadata and dependent GPU commands much more carefully.
For this lesson, correctness and visible semantics matter more than maximum
overlap.

## Why layer lengths can temporarily differ

At the beginning of a decode call, suppose every layer has length 4:

```text
layer 0 length = 4
layer 1 length = 4
layer 2 length = 4
```

The model runs layers in order. After layer 0 calculates and appends the new
token's K/V:

```text
layer 0 length = 5
layer 1 length = 4
layer 2 length = 4
```

This is not an error. Layer 1 has not processed the new hidden state yet.

Later:

```text
after layer 1:

layer 0 length = 5
layer 1 length = 5
layer 2 length = 4
```

After the complete model call:

```text
layer 0 length = 5
layer 1 length = 5
layer 2 length = 5
```

For this reason, `KVCache` stores a separate valid length for every layer:

```cpp
cache.valid_length(layer_index);
```

It also exposes:

```cpp
cache.has_uniform_length();
cache.sequence_length();
```

`sequence_length()` is meaningful only when every layer has the same length.
It throws a logic error if called in the temporary middle-of-forward state.

This protects us from pretending that one global sequence length has already
been committed when only some layers have appended the current chunk.

## Ownership: why `KVCache` cannot be copied

`RuntimeTensor` and `minigpu::Buffer` behave like shared storage handles.
Copying a handle does not duplicate all the GPU bytes:

```text
handle A ──┐
           ├── one Metal allocation
handle B ──┘
```

Shared handles are useful. For example, a queued Metal dispatch can keep a
buffer allocation alive even if the caller's local handle disappears.

But copying a complete mutable `KVCache` would create a dangerous situation:

```text
cache A and cache B share the same K/V bytes
cache A tracks one set of valid lengths
cache B tracks another set of valid lengths
```

An append through `cache A` would modify bytes visible through `cache B`,
while `cache B` might still claim a different logical length.

Therefore the high-level cache has:

```cpp
KVCache(const KVCache&) = delete;
KVCache& operator=(const KVCache&) = delete;
```

It can still be moved:

```cpp
KVCache(KVCache&&) noexcept = default;
```

The design is:

```text
shared low-level buffer lifetime
exclusive high-level cache mutation and metadata
```

Those ideas do not conflict. They solve different problems.

## Resetting does not erase GPU bytes

`reset()` sets every layer's valid length to zero:

```text
before reset:
capacity:   [0] [1] [2] [3] [4] [5]
valid:       ✓   ✓   ✓   ✓   ✗   ✗

after reset:
capacity:   [0] [1] [2] [3] [4] [5]
valid:       ✗   ✗   ✗   ✗   ✗   ✗
```

The old float bit patterns may still physically exist in the allocation.
They are no longer logically readable through `load_key()` or
`load_value()`, because those methods check the valid length.

The next append starts at token position zero and overwrites rows before
making them valid again.

This avoids:

```text
clearing a large buffer
releasing it
allocating it again
```

So:

```text
reset means "forget all valid rows"
reset does not mean "fill every byte with zero"
```

## Memory cost

One layer owns:

```text
one K tensor [Hkv, C, D]
one V tensor [Hkv, C, D]
```

For `L` layers, the total is:

```text
elements =
    2 × L × Hkv × C × D

bytes =
    2 × L × Hkv × C × D × sizeof(element)
```

The factor `2` means:

```text
K + V
```

Lesson 38 stores `float`, so:

```text
sizeof(float) = 4 bytes
```

The helper:

```cpp
calculate_kv_cache_memory_usage(config)
```

reports:

```text
elements and bytes per K or V tensor
elements and bytes per layer
total elements and bytes
```

It also performs overflow and Metal-index-size checks before allocating.

## Memory example for our tiny Hugging Face Llama

The model configuration contains:

```text
L   = 2 layers
Hkv = 4 key/value heads
C   = 2048 maximum positions

hidden size = 16
query heads = 4
D = hidden size / query heads = 16 / 4 = 4
```

For one K tensor in one layer:

```text
4 heads × 2048 tokens × 4 components
= 32,768 floats
= 131,072 bytes
= 128 KiB
```

K and V together for one layer:

```text
256 KiB
```

Both layers:

```text
2 × 2 × 4 × 2048 × 4 × 4 bytes
= 524,288 bytes
= 512 KiB
```

The model is intentionally tiny, so this is small. Real models can have many
more layers, wider heads, more simultaneous requests and long contexts.
KV-cache memory therefore becomes a major inference-engine resource.

The important scaling rule is:

```text
double the capacity → double the KV-cache memory
double the batch/request count → roughly double the cache memory
```

## Multi-head attention versus grouped-query attention

Let:

```text
Hq  = query-head count
Hkv = key/value-head count
```

### Ordinary multi-head attention

In ordinary multi-head attention:

```text
Hq = Hkv
```

Our tiny model uses:

```text
Hq  = 4
Hkv = 4
```

### Grouped-query attention

In grouped-query attention:

```text
Hkv < Hq
```

Several query heads share one key/value head.

For example:

```text
Hq  = 8
Hkv = 2
```

The cache needs only two sets of K/V heads, not eight. That is why the memory
formula uses:

```text
Hkv
```

rather than the query-head count.

Lesson 38's storage configuration already distinguishes
`key_value_head_count`. Lesson 39 will consume the current model's ordinary
multi-head layout, where `Hq == Hkv`. A later grouped-query-attention
extension must map each query head to its corresponding KV head.

## Intended RoPE boundary

In Llama attention:

```text
Q and K projections
        ↓
apply RoPE to Q and K
        ↓
attention
```

The keys retained for future attention should be the position-encoded,
post-RoPE keys. Values are not rotated.

Lesson 38 itself does not perform projections or RoPE. It faithfully stores
the `keys` and `values` tensors supplied by its caller.

Lesson 39 must therefore connect the operations in the correct order:

```text
K projection → RoPE → append rotated K
V projection ───────→ append V
```

## The public API

Create a configuration:

```cpp
const minitensor::inference::KVCacheConfig config{
    .layer_count = 2,
    .key_value_head_count = 4,
    .capacity = 16,
    .head_dimension = 4,
};
```

Allocate the cache:

```cpp
minitensor::inference::KVCache cache(
    device,
    config
);
```

Inspect its state:

```cpp
cache.config();
cache.memory_usage();
cache.valid_length(layer);
cache.has_uniform_length();
cache.sequence_length();
```

Append one layer's compact K/V chunk:

```cpp
cache.append(
    *queue,
    layer,
    new_keys,
    new_values
);
```

Inspect teaching values:

```cpp
cache.load_key(
    layer,
    head,
    token,
    component
);

cache.load_value(
    layer,
    head,
    token,
    component
);
```

Reset all layer lengths:

```cpp
cache.reset();
```

Roll every layer back to an earlier valid prefix:

```cpp
cache.truncate(previous_length);
```

`truncate()` is useful when a higher-level model pass fails after only some
layers have appended. Like `reset()`, it changes logical lengths without
zero-filling the underlying Metal buffers.

The `load_*` methods are useful for tests and learning. A real attention
kernel should read the underlying GPU tensors directly:

```cpp
cache.key_storage(layer);
cache.value_storage(layer);
```

It should not copy every value back through CPU accessor calls.

## Validation performed before an append

`append()` checks:

```text
the layer index exists
threads per group is nonzero
K and V both have rank 3
K shape is [Hkv, new_tokens, D]
V shape is [Hkv, new_tokens, D]
K and V shapes are identical
K and V use compact row-major storage with offset zero
source K/V buffers do not alias the destination layer's cache buffers
the append fits inside capacity
Metal indexes fit the lesson's 32-bit indexing model
```

Compact source tensors keep the first append kernel easy to understand.
A non-contiguous view must first be materialized using the earlier
contiguous-materialization lesson.

The non-aliasing rule prevents a source element from being overwritten by
another GPU lane before it has been read. Normal attention projections
already produce separate output tensors, so Lesson 39 naturally satisfies
this rule.

## What Lesson 38 deliberately does not do

The existence of populated GPU buffers does not automatically change model
execution.

The current model still does:

```text
forward(complete prefix)
```

and its attention still calculates Q, K and V for all prefix positions.

Lesson 38 does **not** yet provide:

```text
one-token cached model forward
attention over [cached K/V + new K/V]
rectangular decode attention
generator integration
cache-aware exact-output comparison
paged allocation
continuous batching
```

The next connection is:

```text
Lesson 38 KVCache
        ↓
Lesson 39 cached attention and cached generation
```

Keeping this boundary explicit prevents a common mistake:

```text
"A cache class exists"
does not mean
"the model is using the cache"
```

## Files

```text
kv_cache.hpp
    public configuration, memory report and KVCache API

kv_cache.cpp
    allocation, validation, flat indexing and Metal append kernel

tests.cpp
    storage, append, length, reset and error-path checks

demo.cpp
    a small real-Metal prefill/decode append walkthrough

README.md
    this conceptual guide
```

## Recommended reading order

You do not need to read every line.

Read:

1. `KVCacheConfig` in `kv_cache.hpp`.
2. The comments above `valid_length()` and `sequence_length()`.
3. The public `append()` declaration.
4. The Metal function named `append_kv_cache`.
5. The bottom half of `KVCache::append()`, where the dispatch is bound,
   submitted and waited for.
6. `demo.cpp` to see a chunk append followed by a one-token append.

You can initially skim:

```text
overflow-checking helpers
detailed exception-message construction
all individual negative tests
```

Those parts make the component safe, but they are not the central KV-cache
idea.

## Build

Run the following commands from the workspace root.

### Build the tests

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
  outputs/infer-lab/phase0/lesson38-kv-cache-layout/kv_cache.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson38-kv-cache-layout/tests.cpp \
  -framework Foundation \
  -framework Metal \
  -o outputs/infer-lab/phase0/work/lesson38_tests
```

Run:

```bash
outputs/infer-lab/phase0/work/lesson38_tests
```

### Build the demonstration

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
  outputs/infer-lab/phase0/lesson38-kv-cache-layout/kv_cache.cpp \
  outputs/infer-lab/phase0/lesson14-metal-minigpu/minigpu.mm \
  outputs/infer-lab/phase0/lesson38-kv-cache-layout/demo.cpp \
  -framework Foundation \
  -framework Metal \
  -o outputs/infer-lab/phase0/work/lesson38_demo
```

## Run

```bash
outputs/infer-lab/phase0/work/lesson38_demo
```

The demonstration uses the Apple GPU. It first appends a multi-token prefill
chunk, then appends a one-token decode chunk, and prints the cache lengths,
selected K/V values and memory accounting.

## Main takeaways

1. A KV cache stores each layer's past **keys and values**.
2. It does not normally store past queries or context vectors.
3. Each layer owns K and V storage shaped `[Hkv, capacity, D]`.
4. Capacity is allocated space; valid length is usable sequence data.
5. Prefill appends many token rows, while decode normally appends one.
6. Preallocation avoids reallocating and copying the entire history on every
   generated token.
7. The Metal append kernel maps compact source rows into the next reserved
   destination rows.
8. Layer lengths may temporarily differ while a model forward processes
   layers one by one.
9. The high-level cache is move-only so two cache objects cannot share bytes
   while disagreeing about mutable lengths.
10. Resetting forgets the rows logically; it does not erase the old bytes.
11. KV-cache memory grows linearly with layers, KV heads, capacity and head
    dimension.
12. Lesson 38 stores the cache, but Lesson 39 must make attention and
    generation consume it.
