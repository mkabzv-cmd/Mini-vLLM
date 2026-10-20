# Inference Lab — Future Roadmap After Lesson 39

## Status

Development is paused after Lesson 39.

The current completed checkpoint is:

```text
real Hugging Face Llama weights
        ↓
real Metal transformer execution
        ↓
prompt prefill
        ↓
per-layer KV cache
        ↓
one-token cached decoding
        ↓
autoregressive text generation
```

Lesson 39 was verified against the earlier full-recomputation path:

```text
same model
same prompt
same generated token IDs
```

The next implementation lesson is:

```text
Lesson 40 — request and sequence lifecycle
```

Everything described after Lesson 39 in this document is a plan. It has not
been implemented yet.

## Quick map

```text
Lessons 40–49
    Mini-vLLM correctness: requests, paging, batching and scheduling

Lessons 50–65
    measurement, async execution, memory, precision and optimized kernels

Lessons 66–68
    transformer compiler integration

Lessons 69–73
    prefix caching, speculation, GPU sampling, profiling and HTTP

Lessons 74–89
    Mini-Dynamo and NVIDIA specialization
```

## How we will work when development resumes

Future lessons should continue one at a time.

For each lesson:

1. Explain the central idea without code.
2. Use one small numerical or memory-layout example.
3. Separate CPU/runtime responsibilities from GPU-kernel responsibilities.
4. Identify the important invariant.
5. Implement only the new layer required by that lesson.
6. Compare its results with the previous correct implementation.
7. Tour only the most important code afterward.

We should not generate the entire roadmap in one implementation pass.

The purpose is to understand why each layer exists, not merely to accumulate
files.

## The hierarchy we are building

The complete project will eventually look approximately like this:

```text
streaming inference API
        ↓
Mini-Dynamo distributed coordination
        ↓
Mini-vLLM request scheduler and KV-cache manager
        ↓
Llama model executor
        ↓
transformer operators and compiler
        ↓
MiniTensor
        ↓
MiniGPU / Metal runtime
        ↓
Apple GPU
```

Each layer solves a different problem.

```text
MiniGPU
    How do we submit work and manage GPU-accessible memory?

MiniTensor
    How do tensors, shapes, strides, views and operations behave?

Compiler
    How do we validate, optimize, fuse and lower tensor programs?

Model executor
    How do we run one transformer model correctly?

Mini-vLLM
    How do we serve many sequences efficiently on one worker?

Mini-Dynamo
    How do we coordinate many workers, devices or machines?
```

---

# Stage 1 — Lessons 40–49: Mini-vLLM

## Goal

Lesson 39 runs one request with one contiguous KV cache.

By Lesson 49, the engine should support:

```text
multiple concurrent requests
shared paged KV-cache memory
per-sequence block tables
memory-aware scheduling
batched prefill
batched decode
continuous joining and leaving
cancellation and resource reclamation
correct output parity with Lesson 39
```

The first Mini-vLLM version will remain synchronous. It may still wait at
many GPU boundaries. Correct request scheduling and memory ownership come
before asynchronous optimization.

## Important terminology

Two unrelated concepts both use the word `block`:

```text
GPU threadgroup
    a group of lanes executing a kernel together

KV-cache block/page
    a fixed-size region of memory holding several token positions
```

A KV page is not a GPU threadgroup, and it does not imply disk storage.

---

## Lesson 40 — Requests, sequences and lifecycle

### Question answered

How does a user generation job live across many engine iterations?

Lesson 39 immediately receives one prompt and runs it to completion. A
serving engine instead needs an object that can:

```text
wait
prefill
decode over many steps
finish
fail
or be cancelled
```

### Concepts

Initially:

```text
one request → one sequence
```

A request represents the user's API-level job and sampling configuration.

A sequence represents one concrete token history being executed. They begin
one-to-one, but keeping them separate matters because a later algorithm such
as beam search or speculative decoding may attach several candidate
sequences to one request.

A sequence contains:

```text
prompt token IDs
generated token IDs
number of tokens already consumed by the model
the uncomputed token suffix
sampling configuration
per-request random seed/state or counter
status
finish reason
```

During ordinary one-token decode, the uncomputed suffix is usually one token.

The lifecycle is:

```text
Waiting
   ↓
Prefilling
   ↓
Decoding
   ├──→ Finished: EOS
   ├──→ Finished: length limit
   ├──→ Cancelled
   └──→ Failed
```

### What we build

Small CPU-side types:

```cpp
RequestId
SamplingParams
SequenceStatus
FinishReason
Sequence
Request
```

No new Metal kernel is required.

### Important invariant

We must distinguish:

```text
tokens present in the sequence
versus
tokens already processed into the KV cache
```

The newest sampled token may already be emitted to the user but not yet have
been consumed by the next decode step.

We must also enforce:

```text
prompt positions + generated positions
    ≤ model maximum context length
```

The model context limit is different from the current number of free blocks
in the shared pool. A request may be mathematically too long even when memory
is available, or temporarily unschedulable because the pool is full even
though it remains within the model limit.

### Verification

- Legal and illegal state transitions.
- EOS completion.
- Maximum-token completion.
- Cancellation while waiting or running.
- Duplicate request-ID rejection.
- Exact token-history accounting.

### Production connection

These objects become the contract between:

```text
API frontend
scheduler
model executor
output streamer
```

---

## Lesson 41 — Shared KV-block pool

### Question answered

How can many requests share one reserved region of KV-cache memory without
allocating one maximum-sized contiguous buffer per request?

### Why the Lesson 39 design is insufficient

Lesson 39 allocates approximately:

```text
one request → one contiguous cache at its requested capacity
```

If a request reserves 4,096 positions but uses only 80, most of that
allocation is wasted.

### Central idea

Divide a large GPU allocation into fixed-size KV blocks.

For teaching, a block might hold four token positions:

```text
block size = 4 token positions

physical block 0
physical block 1
physical block 2
...
```

The pool owns K and V storage conceptually shaped like:

```text
K[layer, physical_block, kv_head, slot, component]
V[layer, physical_block, kv_head, slot, component]
```

Here:

```text
slot = token position inside one block
```

For this miniature design, one `BlockId` reserves the corresponding K and V
slots across every transformer layer. The pool belongs to exactly one model,
cache layout and dtype. A cache or block ID must not cross into another
engine merely because its dimensions happen to match.

### What we build

```cpp
class KVBlockPool {
public:
    BlockId allocate();
    void release(BlockId);
    std::size_t free_block_count() const;
};
```

The CPU manages the free list. K/V payloads remain in Metal buffers.

### Important invariant

One physical block has exactly one owner unless a later feature explicitly
allows safe sharing.

### Verification

- Allocations return unique IDs.
- Exhaustion fails without corrupting state.
- Released blocks can be reused.
- Invalid release and double-free are rejected.
- Memory accounting matches buffer sizes.
- Every test finishes with all blocks returned.

---

## Lesson 42 — Per-sequence block tables and paged append

### Question answered

How does a sequence appear logically contiguous when its KV blocks are
physically scattered?

### Block table

A sequence may own:

```text
logical block     0  1  2
physical block   [7, 2, 9]
```

With four token positions per block, token position 6 maps as:

```text
logical block = 6 / 4 = 1
slot          = 6 % 4 = 2
physical      = block_table[1] = 2
```

The sequence still sees positions:

```text
0, 1, 2, 3, 4, 5, 6...
```

The runtime translates them into physical locations.

### What we build

```cpp
class PagedKVCache {
    KVBlockPool* pool;
    std::vector<BlockId> block_table;
    std::size_t valid_length;
};
```

Operations include:

```text
reserve enough blocks
append new K/V
truncate after failure
release all owned blocks
```

The Metal append kernel receives the block table and maps:

```text
logical token position
        ↓
physical block and slot
        ↓
flat GPU-buffer address
```

### Important invariant

The valid length must advance only after:

```text
required blocks are successfully reserved
and
the complete model step succeeds
```

KV append occurs partway through a layer. Attention, later layers or the
final logits projection can still fail. New lengths and newly allocated
blocks therefore remain pending until the entire prefill/decode transaction
commits. Failure restores the old lengths and returns new blocks.

### Verification

- Append across a block boundary.
- Use deliberately scrambled physical IDs.
- Reconstruct logical K/V and compare with Lesson 39.
- Truncate and release unused trailing blocks.
- Roll back ownership after an injected failure.
- Preserve a partially filled final block.

---

## Lesson 43 — Paged attention

### Question answered

How can attention read K/V when historical tokens do not occupy one
contiguous physical buffer?

### Central idea

For each logical cached token:

```text
token position
     ↓
logical block and slot
     ↓ block table
physical block
     ↓
K/V address
```

The attention mathematics does not change.

Only address calculation changes.

### What we build

Paged attention receives:

```text
shared K/V pool buffers
one sequence's block table
context length
query start position
```

Initially it handles one request per dispatch. This isolates the new idea:
paged address translation.

### Important invariant

Logical token ordering must be independent of physical block ordering.

### Verification

Compare:

```text
CPU attention reference
Lesson 39 contiguous cached attention
Lesson 43 paged attention
```

Test:

- One block.
- Several blocks.
- Scrambled physical block IDs.
- Partial final block.
- Block-boundary positions.
- Causal masking using absolute query positions.
- Separately verify that the executor applies RoPE before K is written into
  paged storage.

---

## Lesson 44 — Memory-aware scheduling

### Question answered

Which waiting and running requests should execute in the next engine step?

### Inputs to the decision

```text
waiting requests
running decode requests
free KV blocks
maximum sequences per batch
maximum scheduled tokens
prefill chunk size
```

### First scheduling policy

Keep the first policy understandable:

1. Allow active decode requests to make progress.
2. Use remaining token budget for FIFO prefill chunks.
3. Reserve the required KV blocks transactionally before committing the
   plan.
4. Leave work waiting if memory is unavailable.
5. Reserve some capacity for aged prefill work so a steady stream of decode
   requests cannot starve waiting prompts forever.
6. Do not implement preemption yet.

### What we build

```cpp
struct SchedulerPlan {
    std::vector<PrefillWork> prefills;
    std::vector<DecodeWork> decodes;
    std::size_t scheduled_tokens;
};
```

The lesson creates plans but does not execute the model.

### Important invariant

The plan may never promise more:

```text
tokens
sequences
or KV blocks
```

than the engine actually owns.

The scheduler and block manager are the only components that reserve physical
KV slots. A committed plan contains those reservations. The model executor
only consumes the supplied plan, while the engine commits or rolls back the
reservations after execution.

Prompts exceeding the model context limit, or requests that could never fit
even in an otherwise empty pool, are rejected rather than waiting forever.

### Verification

- Deterministic FIFO behavior.
- Token and sequence budgets are respected.
- Decode requests continue progressing.
- Long prompts become several chunks.
- Memory exhaustion leaves requests waiting safely.
- Cancellation and completion release reservations.

---

## Lesson 45 — Packed batch metadata

### Question answered

How do variable-length requests become one GPU-friendly batch?

### Why simple padding is wasteful

Suppose:

```text
request A contributes 3 token rows
request B contributes 1 token row
```

Instead of padding B to three rows:

```text
packed batch = 4 real rows
```

Every packed row needs metadata:

```text
request/sequence ID
input token ID
absolute position
destination KV location
```

Every sequence needs metadata:

```text
range of packed query rows
context length
block-table offset
block count
```

### What we build

```cpp
PackedModelBatch
BatchMetadata
ModelBatchBuilder
```

All participating block tables can be flattened:

```text
[A blocks..., B blocks..., C blocks...]
```

Per-sequence offsets locate each subtable.

### Important invariant

Packing may change physical row order, but it must never lose the mapping
back to the owning sequence.

### Verification

- Variable-length rows pack without padding.
- CPU metadata reconstructs the original work exactly.
- A diagnostic Metal kernel reports the same row mappings.
- Reordered requests still receive the correct outputs.
- Oversized and malformed plans are rejected.

---

## Lesson 46 — Batched and chunked prefill

### Question answered

How can several prompt chunks run in one model execution?

### Example

```text
request A → 3 new prompt tokens
request B → 2 new prompt tokens
request C → 1 new prompt token
```

Shared transformer projections can operate on:

```text
packed hidden states [6, hidden_size]
```

Attention must still isolate requests:

```text
A may attend only to A's history
B may attend only to B's history
C may attend only to C's history
```

### Chunked prefill

A long prompt does not have to monopolize one enormous engine iteration.

```text
1,000-token prompt
    ↓
several smaller scheduled chunks
```

Sampling begins only after the prompt is fully processed.

### What we build

A paged batched executor that:

1. Embeds packed rows.
2. Runs shared transformer projections.
3. Writes each sequence into the physical slots already reserved by the
   scheduler plan.
4. Performs segmented paged attention.
5. Returns logits only for sequences completing prefill.

### Important invariant

One request must never read another request's K/V.

### Verification

- Compare each sequence with independent Lesson 39 execution.
- Different prompt lengths.
- Multiple prompt chunks.
- Physical block-boundary crossings.
- Failure injection with transactional rollback for every affected sequence.

---

## Lesson 47 — Batched decode

### Question answered

How can one new token from many sequences run in one model pass?

### Shapes

For `B` decoding sequences:

```text
input hidden states [B, hidden_size]
logits              [B, vocabulary_size]
```

Each row has:

```text
one pending input token
one absolute position
its own context length
its own block table
```

The linear layers benefit naturally because `B` becomes their matrix row
count.

### What we build

```cpp
PagedLlamaExecutor::decode_batch(...)
```

It:

1. Packs one pending token per sequence.
2. Uses the physical cache positions already reserved in the scheduler plan.
3. Appends one K/V position per sequence and layer.
4. Runs paged attention with different context lengths.
5. Returns one logits row for each sequence.

### Important invariant

Batch-row order and request-ID order are not assumed to be identical.
Explicit metadata maps outputs back to requests.

### Verification

- Logits match separate Lesson 39 decodes within a defined tolerance.
- Greedy token IDs match exactly for selected deterministic fixtures.
- Different context lengths.
- Some sequences cross a block boundary while others do not.
- Reordered batch rows.
- Greedy and seeded sampling parity.
- No cross-request cache contamination.

---

## Lesson 48 — Continuous batching engine

### Question answered

How do requests join and leave while other requests continue generating?

### One engine iteration

```text
accept new requests
        ↓
scheduler creates a plan
        ↓
run prefill and/or decode batches
        ↓
sample tokens
        ↓
emit token events
        ↓
finish/cancel and reclaim resources
```

Continuous batching means that an original fixed batch does not have to
remain together until every request finishes.

### What we build

```cpp
class MiniEngine {
public:
    RequestId submit(...);
    void cancel(RequestId);
    StepResult step();
    bool has_unfinished_requests() const;
    std::vector<TokenEvent> take_outputs();
};
```

Execution remains synchronous for teaching clarity.

### Timeline example

```text
step 0: request A enters
step 1: request B enters while A continues
step 2: B reaches EOS and releases its blocks
step 3: request C enters while A continues
```

### Verification

- Correct outputs for overlapping requests.
- Fair progress.
- Block reclamation at the next safe boundary after submitted GPU work no
  longer references the blocks. In this synchronous version, that is after
  the current engine step finishes.
- Cancellation.
- No leaked blocks.
- A pre-dispatch validation failure can fail one request independently.
- A command-buffer or device failure may require rolling back or failing the
  complete affected batch.

---

## Lesson 49 — Complete Mini-vLLM

### Goal

Integrate the Stage 1 pieces behind one small public API:

```cpp
MiniVllmEngine engine(model, config);

RequestId id = engine.submit(prompt, sampling);

while (engine.has_unfinished_requests()) {
    engine.step();

    for (TokenEvent event : engine.take_outputs()) {
        // In-process token streaming.
    }
}
```

### Configuration

```text
KV block size
number of physical blocks
maximum batch tokens
maximum active sequences
prefill chunk size
sampling defaults
```

### Initial counters

```text
prompt tokens processed
decode tokens processed
engine iterations
prefill and decode batch sizes
allocated and free blocks
peak cache utilization
finished, cancelled and failed requests
```

### Completion test

- Several simultaneous real Hugging Face prompts.
- Logits match Lesson 39 within a defined tolerance.
- Greedy token IDs match exactly for selected deterministic fixtures.
- Deterministic seeded sampling.
- A tiny block pool stressed through repeated reuse.
- Randomized submit/cancel/finish sequences.
- Static analysis and sanitizers.
- Real Metal execution.
- Every test ends with:

```text
free blocks == total blocks
```

### What Mini-vLLM will not yet claim

```text
asynchronous GPU overlap
mixed precision
optimized production kernels
FlashAttention
prefix sharing
speculative decoding
quantization
HTTP serving
multi-device execution
```

Those belong to the next stage.

---

# Stage 2 — Lessons 50–73: Optimized serving engine

## Goal

Lesson 49 should work correctly, but much of it will still be intentionally
inefficient.

This stage reduces:

```text
CPU waiting
GPU dispatch overhead
temporary allocations
memory traffic
KV-cache waste
weight memory
time per output token
```

Correctness expectations depend on the kind of change:

```text
scheduler, paging, asynchronous execution and memory reuse
    same mathematical path
    tight logit tolerance
    exact greedy fixture tokens

mixed precision and numerically equivalent fusion
    explicitly chosen logit tolerance
    fixture-level generation checks

quantization
    bounded numerical/quality degradation
    memory and speed improvement
    exact token identity is not always expected
```

---

## Lesson 50 — Profiling foundations and a trusted baseline

### Why this comes first

We should measure before optimizing.

Otherwise, we may make code more complicated without knowing whether the
complete system improved.

### Metrics

```text
TTFT
    time from request arrival to its first generated token

TPOT
    average time per output token after the first

ITL
    time between individual streamed tokens

latency
    total request time

throughput
    aggregate input/output tokens processed per second
```

Memory metrics:

```text
model-weight memory
KV-cache memory
temporary/workspace memory
allocator high-water mark
free versus occupied KV blocks
```

Scheduler metrics:

```text
queue time
prefill batch size
decode batch size
scheduled tokens per iteration
cache-pressure stalls
```

### What we build

Timestamped events:

```text
request submitted
request scheduled
prefill started and finished
decode iteration started and finished
first token emitted
request completed
```

GPU timing must include correct synchronization. Timing only CPU submission
would measure how quickly commands were queued, not how long the GPU took.

Benchmark methodology must remain controlled:

```text
warm up model and kernels
run repeated synchronized trials
fix prompt lengths and requested output lengths
fix concurrency and scheduler limits
separate cold start from steady state
report input-token/s and output-token/s separately
report p50, p95 and p99 rather than one lucky run
```

---

## Lesson 51 — Asynchronous execution

### Current problem

Many teaching operators approximately do:

```text
submit kernel
wait
submit kernel
wait
submit kernel
wait
```

This repeatedly blocks the CPU even when later commands could already be
queued.

### Desired execution

```text
CPU submits RMSNorm
CPU submits QKV projections
CPU submits RoPE
CPU submits attention
CPU submits output projection and feed-forward work
        ↓
GPU queue executes dependencies in order
        ↓
CPU waits only at a genuine host-visible boundary
```

### What we build

```text
asynchronous operator submission
events/fences representing unfinished work
deferred error propagation
dependency tracking
buffer lifetime tracking
explicit host synchronization points
one command batch for a model iteration
```

A tensor may mean:

```text
storage has been allocated
producer work has been submitted
contents are not yet CPU-readable
```

Cancellation also becomes asynchronous. A cancelled request may stop being
scheduled immediately, but its KV blocks and pooled temporary buffers cannot
be returned while already-submitted GPU commands still reference them.
Reclamation is deferred until the request's final in-flight event completes.

### Important limitation

Autoregressive generation still has a token dependency:

```text
decode logits
    ↓
select next token
    ↓
submit its next decode
```

We can remove most waits inside a step even though a dependency remains
between tokens.

### Transfer to CUDA

```text
Metal command queues and events
        ↔
CUDA streams and events
```

---

## Lesson 52 — Temporary-buffer reuse and memory pools

### Current problem

Without reuse:

```text
RMSNorm       → allocate
Q projection  → allocate
K projection  → allocate
attention     → allocate
SwiGLU        → allocate
```

Repeated allocation adds overhead and fragments memory.

### Memory categories

Persistent:

```text
model weights
KV-cache blocks
request state
```

Temporary:

```text
normalized activations
Q/K/V intermediates
attention workspaces
matmul scratch space
```

### What we build

```text
GPU buffer pool
size classes
acquire and release API
per-batch scratch arena
event-aware safe reuse
memory accounting
high-water marks
```

### Important invariant

A C++ handle disappearing does not prove that the GPU finished using its
storage.

The pool may recycle a buffer only after the relevant GPU event completes.

### Compiler connection

Later, compiler liveness analysis can prove:

```text
temporary A's final use occurs before temporary B begins
```

Then B may reuse A's allocation.

---

## Lesson 53 — FP16 and BF16 inference

### Why lower precision matters

```text
FP32 = 4 bytes per number
FP16 = 2 bytes per number
BF16 = 2 bytes per number
```

Reducing bytes can lower:

```text
weight memory
activation memory
KV-cache memory
memory bandwidth
```

### Storage precision versus accumulation precision

A common policy is:

```text
weights              FP16 or BF16
activations          FP16 or BF16
matmul accumulation  FP32
softmax statistics   FP32
RMSNorm statistics   FP32
```

### FP16 versus BF16

```text
FP16
    more fraction precision in its normal range
    smaller exponent range

BF16
    fewer fraction bits
    exponent range similar to FP32
```

### What we build

```text
dtype stored in RuntimeTensor
typed allocation and binding
conversion kernels
mixed-precision validation
FP32 accumulation paths
numerical error-tolerance tests
single-file FP16 loading or conversion
hardware-gated BF16 support
```

---

## Lesson 54 — MHA, GQA and MQA

### Why this is needed

The current simple attention path largely assumes one K/V head for each
query head.

Modern models may use fewer K/V heads.

```text
MHA:
    8 query heads
    8 KV heads

GQA:
    8 query heads
    2 KV heads

MQA:
    8 query heads
    1 KV head
```

For GQA:

```text
query heads 0–3 → KV head 0
query heads 4–7 → KV head 1
```

### Why models do this

Decode repeatedly reads the KV cache.

Fewer KV heads mean:

```text
smaller cache
less memory traffic
more request capacity
```

### What we change

```text
model configuration validation
Q projection width = query_heads × head_dimension
K projection width = KV_heads × head_dimension
V projection width = KV_heads × head_dimension
Q-head-to-KV-head mapping
cache layouts
attention kernels
RoPE shapes
prefill/decode tests
```

Important rule:

```text
query_head_count % kv_head_count == 0
```

For one query head:

```text
kv_head =
    query_head / (query_head_count / kv_head_count)
```

---

## Lesson 55 — Sharded checkpoints and meaningful trained models

### Current limitation

Our tiny random model is excellent for correctness testing but produces
meaningless text.

Real checkpoints may be split:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...
model.safetensors.index.json
```

The index maps each tensor name to a shard.

### What we build

```text
index-file parser
tensor-to-shard lookup
loading without first merging every shard
extension of Lesson 53's typed loader across multiple shards
configuration validation
tokenizer compatibility checks
useful missing-tensor and shape diagnostics
optional lazy loading
```

Then run a genuinely trained small model and compare:

```text
selected logits
greedy token IDs
generated text
```

with a trusted reference implementation.

---

## Lesson 56 — Proper optimized tiled matmul and linear kernels

### Current limitation

Our lessons teach correct matmul and a basic tiled version, but they do not
yet provide a production-quality family of kernels.

One kernel shape is not ideal for every workload.

```text
prefill:
    many matrix rows

decode:
    often M = 1 or a small batch
```

### What we build

```text
two-dimensional threadgroups
cooperative A/B tile loading
threadgroup memory
barriers
register accumulators
irregular-boundary handling
separate prefill and decode paths
benchmarks against current kernels
```

The decode path may resemble matrix-vector or small-M matrix multiplication,
while prefill resembles a larger GEMM.

---

## Lesson 57 — Vectorization, matrix instructions and kernel selection

After correct tiling:

```text
vectorized loads/stores
packed half-precision data
multiple outputs per lane
SIMD-group matrix operations
weight packing/layout transformation
several tile configurations
```

The runtime selects a kernel using:

```text
matrix dimensions
dtype
prefill versus decode
GPU limits
```

A small autotuner can benchmark candidates and cache its choice.

### Main lesson

There is no universally fastest matmul kernel.

Production libraries maintain several implementations specialized for
different shapes, types and hardware.

---

## Lesson 58 — Fused residual addition and RMSNorm

### Why fusion helps

Without fusion:

```text
residual add
    ↓ write temporary
RMSNorm
    ↓ reread temporary
write normalized output
```

With fusion:

```text
read activation and residual
add
calculate normalization
write one final result
```

### What we build

```text
unfused reference path
fused residual-add-plus-RMSNorm kernel
FP32 norm accumulation
numerical comparison
dispatch-count comparison
temporary-memory comparison
```

---

## Lesson 59 — Fused QKV, RoPE and cache writing

### Current path

```text
Q projection
K projection
V projection
RoPE on Q
RoPE on K
separate KV append dispatch
```

### Desired path

```text
hidden state
    ↓ combined QKV projection
Q → RoPE → attention
K → RoPE → final KV-cache address
V ─────────→ final KV-cache address
```

### What we investigate

```text
combined QKV weight layout
fewer projection launches
RoPE while writing Q/K
direct cache writes
eliminating the standalone append dispatch
```

This lesson connects directly to the earlier observation that even simple KV
copying currently requires its own GPU dispatch.

---

## Lesson 60 — Fused SwiGLU

A Llama feed-forward block computes:

```text
gate = x × W_gateᵀ
up   = x × W_upᵀ

hidden = SiLU(gate) × up

output = hidden × W_downᵀ
```

Possible fusion:

```text
combined gate/up projection
        ↓
SiLU and multiply without two long-lived intermediates
        ↓
down projection
```

### What we build

```text
combined gate/up weight representation
combined projection
fused SiLU-and-multiply kernel
temporary-memory accounting
parity with the unfused path
```

---

## Lesson 61 — Online softmax

### Current attention structure

```text
scores = Q × Kᵀ
probabilities = softmax(scores)
context = probabilities × V
```

This may materialize complete score and probability tensors.

### Online idea

Process keys in blocks while maintaining:

```text
running maximum
running exponential sum
running weighted-V result
```

If a later block has a larger maximum, earlier accumulated values are
rescaled correctly.

### What we build

1. A tiny numerical CPU example.
2. A CPU reference implementation.
3. A simple Metal implementation.
4. Tests against ordinary stable softmax.

### Main result

Stable attention can be computed without retaining the complete score matrix.

---

## Lesson 62 — FlashAttention foundations for contiguous prefill

FlashAttention combines:

```text
QK score calculation
causal masking
online softmax
weighted-V accumulation
```

inside tiled kernels.

Instead of:

```text
write full scores
read scores
write probabilities
read probabilities
```

we do:

```text
load Q/K/V tiles
calculate partial scores
update online softmax
accumulate context
```

The first implementation deliberately handles contiguous prefill:

```text
many query rows
many key rows
one sequence
contiguous Q/K/V tensors
causal masking
```

### Verification

- Causal correctness.
- Absolute query positions.
- FP32 online-softmax statistics.
- Logit/context tolerance against ordinary attention.
- No full score/probability tensor.
- Temporary-memory and timing comparisons.

This lesson teaches the tiled online algorithm without also introducing
paged addressing and ragged batches.

---

## Lesson 63 — Paged decode attention and tuning

Decode has a different shape:

```text
one query per active sequence
different context lengths
scattered paged K/V blocks
MHA, GQA or MQA head mapping
```

This lesson integrates online attention with:

```text
block tables
ragged sequence metadata
paged K/V loads
query-head-to-KV-head mapping
decode-oriented tile sizes
```

### Verification

- Compare with Lesson 43 paged attention.
- MHA/GQA/MQA.
- Different context lengths in one batch.
- Scrambled physical block IDs.
- Partial final blocks.
- Numerical tolerance and timing.
- Tune prefill and decode independently.

---

## Lesson 64 — Quantization foundations

### Idea

Store approximate weights using fewer bits:

```text
FP16 → 16 bits
INT8 → 8 bits
INT4 → 4 bits
```

An integer is interpreted with scale information:

```text
approximate weight = quantized integer × scale
```

Some formats also use a zero point.

### Concepts

```text
symmetric versus asymmetric quantization
per-tensor scales
per-output-channel scales
group-wise scales
group size
zero points
packing two INT4 values into one byte
accuracy versus memory
```

### First implementation

```text
FP32 reference weights
        ↓ quantize
INT8 storage + scale
        ↓ dequantize
approximate FP32 values
```

This first lesson focuses on representation and error.

---

## Lesson 65 — Quantized linear and model execution

Quantization becomes useful only when matmul consumes the compressed form
directly.

Expanding the entire weight matrix back to FP32 before every multiplication
would lose much of the memory and bandwidth benefit.

### What we build

Progressively:

```text
per-channel INT8 linear
group-wise INT4 linear
packed INT4 loads
dequantization inside the matmul loop
FP32 accumulation
one documented internal quantized format
one explicitly selected external checkpoint format
```

Formats such as GPTQ, AWQ and GGUF use different packing and metadata. We
will not claim generic support for all quantized formats from one loader.

### Verification

```text
weight reconstruction error
linear-output error
selected-logit error
generation quality
memory reduction
speed comparison
```

---

# Stage 2B — Compiler integration

The current compiler handles a real but limited elementwise path.

Most transformer operations are still handwritten eager Metal operators.

The practical next compiler design is:

```text
captured model graph
        ↓
typed tensor IR
        ↓
shape, dtype and layout validation
        ↓
optimization and fusion patterns
        ↓
memory/execution plan
        ↓
optimized-kernel selection or generated MSL
        ↓
Metal dispatches
```

The compiler does not initially need to generate every optimized matmul or
FlashAttention instruction itself. It can recognize operations and select
the tuned kernel library we built by hand.

---

## Lesson 66 — Transformer graph capture and IR

The current Llama executor calls eager C++ operators directly. Before the
compiler can optimize a transformer, we need a frontend that records those
calls as a graph instead of immediately executing them.

### What we build

```text
symbolic RuntimeTensor/value handles
graph-building versions of transformer operators
model tracing/capture for prefill and decode
explicit graph inputs for dynamic metadata
compiled-plan cache
```

Dynamic values such as these remain runtime inputs:

```text
batch size
sequence lengths
absolute positions
block tables
physical destination slots
```

The plan cache needs a clear specialization key so we neither recompile every
token nor reuse a plan whose shape/dtype/layout assumptions are invalid.

### Transformer-aware IR

Add operations such as:

```text
Matmul
Linear
RMSNorm
RoPE
Attention
SwiGLU
ResidualAdd
KVCacheRead
KVCacheAppend
```

Each operation carries:

```text
shape
dtype
layout/strides
device
persistent or temporary storage role
side-effect information
```

### Why KV cache is special

Ordinary SSA values behave like immutable mathematical results.

KV append mutates persistent state:

```text
decode step 7 writes position 7
decode step 8 must observe that write
```

The IR must preserve this ordering so an optimizer cannot remove or reorder a
cache write incorrectly.

`KVCacheRead` also participates in the same effect/version ordering. A read
must not move before the append whose new data it is supposed to observe.

---

## Lesson 67 — Shape inference, liveness and execution planning

The compiler determines:

```text
result shapes
legal broadcasting
head mappings
dtypes
layouts
value lifetimes
buffer reuse opportunities
GPU dependencies
host synchronization points
```

Example:

```text
temporary A's last use = operation 5
temporary B begins     = operation 6
```

The execution planner may assign both to the same pooled allocation.

The output of this lesson is an execution plan, not immediate eager
execution.

---

## Lesson 68 — Kernel lowering and automatic transformer fusion

The compiler maps patterns to implementations:

```text
ResidualAdd + RMSNorm
    ↓
fused RMSNorm kernel
```

```text
QKV projection + RoPE + KV append
    ↓
fused QKV/cache kernel
```

```text
Attention with many query rows
    ↓
prefill FlashAttention kernel
```

```text
Attention with one query per sequence
    ↓
paged decode-attention kernel
```

Simple elementwise chains may still generate MSL directly.

Complex operations can select prebuilt tuned kernels.

Later extensions may include:

```text
cost models
kernel templates
autotuning
hardware-specific schedules
```

---

## Lesson 69 — Prefix caching

### Problem

Many requests may share:

```text
the same system prompt
the same document prefix
the same conversation history
```

Without prefix caching, each request repeats the same prefill.

### Central idea

Build a cumulative identity for every complete token block:

```text
block key =
    hash(
        parent-prefix key,
        current block's exact token IDs,
        model/cache identity
    )

block key
    ↓
shared KV block IDs
```

K/V for a block depends on everything before it, so hashing only the current
block's tokens would be incorrect.

### What we build

```text
block hashing
exact token/metadata verification after a hash match
prefix lookup
reference-counted shared blocks
private suffix allocation when sequences diverge
eviction under memory pressure
hit/miss metrics
```

Initially, share only complete immutable blocks. An incomplete tail block
remains private. Copy-on-write is needed only if a later design chooses to
share a partially filled block.

### Cache identity

A safe prefix-cache key must include enough identity:

```text
model and model revision
token sequence
position assumptions
adapter, if any
relevant inference configuration
cache format/dtype
```

Token IDs alone are insufficient if two model executions are incompatible.

---

## Lesson 70 — Speculative decoding

### Idea

A smaller or cheaper draft model proposes several tokens:

```text
draft → A B C D
```

The target model verifies them in one larger pass.

```text
accepted → A B C
rejected → D
```

If several candidates are accepted, one expensive target pass advances
several output positions.

### What we build

```text
draft-model interface
candidate proposal
batched target verification
accept/reject logic
separate draft and target KV caches
checkpoints for both caches
cache truncation after rejected candidates
tokenizer and vocabulary compatibility validation
greedy correctness first
sampling-correct version later
```

The `truncate()` concept introduced around Lesson 39 becomes important here.

If every proposed token is accepted, the target model can also produce one
additional target-selected bonus token.

For sampling, correctness requires the accepted-probability rule and
residual-distribution sampling after a rejection; it is not enough to reuse
the greedy accept/reject rule.

### Metrics

```text
candidate acceptance rate
target-model calls
accepted tokens per verification
TTFT/TPOT changes
```

---

## Lesson 71 — GPU-side sampling

### GPU-side sampling

The simple path often does:

```text
GPU produces logits
CPU reads logits
CPU chooses a token
CPU submits the next decode
```

A GPU sampler can perform:

```text
argmax
temperature scaling
top-k
probability sampling
```

and return only the selected token ID.

Each request needs independent deterministic RNG state or a counter-based
generator. Reordering requests inside a batch must not silently change their
random streams.

The GPU path must preserve:

```text
temperature and top-k semantics
EOS handling
per-request seeds/counters
deterministic test mode
mapping selected tokens back to request IDs
```

---

## Lesson 72 — Final performance and capacity study

Repeat the Lesson 50 benchmark matrix:

```text
FP32 versus FP16/BF16
MHA versus GQA
basic versus optimized matmul
unfused versus fused
ordinary versus FlashAttention
FP16 versus quantized weights
prefix cache hit versus miss
normal versus speculative decoding
```

Report:

```text
TTFT
TPOT
ITL
tokens per second
p50/p95/p99 latency
peak memory
KV bytes per token
dispatches per output token
kernel-time breakdown
maximum practical concurrent requests
```

---

## Lesson 73 — Streaming HTTP inference API

### Request path

```text
HTTP request
    ↓
validation and tokenization
    ↓
Mini-vLLM scheduler
    ↓
prefill/decode
    ↓
incremental detokenization
    ↓
stream generated text
```

### What we build

```text
generate endpoint
request IDs
server-sent events or chunked responses
cancellation
timeouts
slow-client backpressure
prompt/generation limits
health endpoint
metrics endpoint
graceful shutdown
```

The server should bind to localhost by default.

The HTTP layer must submit work through the scheduler. It must not bypass the
engine and call the model directly.

### Completion checkpoint

At Lesson 73:

```text
one optimized serving worker
        ↓
accepts concurrent HTTP requests
        ↓
streams generated tokens
        ↓
uses one Mini-vLLM engine internally
```

---

# Stage 3 — Mini-Dynamo distributed inference

## Goal

Mini-vLLM is the inference engine inside one worker:

```text
requests
    ↓
local scheduler
    ↓
model execution
    ↓
local KV-cache manager
    ↓
one device
```

Mini-Dynamo sits above one or more workers:

```text
clients
    ↓
frontend/router
    ↓
global scheduling
    ↓
model workers
    ↓
local Mini-vLLM engines
```

Mini-Dynamo does not replace Mini-vLLM. It coordinates several instances.

The numbering below is provisional because distributed topics may expand
after the single-worker engine is measured.

---

## Lesson 74 — Worker abstraction and process boundary

Define a model worker that owns:

```text
model weights
Mini-vLLM engine
device queues
KV block pool
active sequences
local scheduler
```

Start with fake workers so process/control bugs are not confused with model
bugs.

Worker operations:

```text
load model
accept request
cancel request
stream output
report load
report health
drain
shut down
```

Important ownership rule:

```text
a request normally remains on the worker holding its KV cache
```

Define distributed ownership explicitly:

```text
model deployment
├── replica group A
│   ├── rank 0 worker → device 0
│   └── rank 1 worker → device 1
└── replica group B
    ├── rank 0 worker → device 2
    └── rank 1 worker → device 3
```

One replica group represents one complete logical copy of the model. If
tensor-parallel size is one, the group contains one ordinary full-model
worker. If tensor-parallel size is greater than one, the group's ranks must
execute the request collectively.

The global router chooses a replica group, not an arbitrary tensor-parallel
rank.

---

## Lesson 75 — RPC protocol

Separate processes cannot call ordinary C++ methods directly.

They exchange serialized messages over:

```text
Unix sockets
or local TCP
```

Messages include:

```text
protocol version
operation
request ID
generation epoch
decode step
deadline
payload size
```

We need:

```text
request/response correlation
streamed token events
timeouts
cancellation messages
bounded queues
malformed-message rejection
```

The transport should be replaceable without redesigning the engine.

---

## Lesson 76 — Worker health and lifecycle

Distinguish:

```text
liveness
    is the process responding?

readiness
    can it currently accept inference work?
```

A worker may be alive but not ready while:

```text
loading weights
warming kernels
recovering memory
draining before shutdown
```

Build:

```text
worker registration
heartbeats
last-seen timestamps
healthy/unhealthy/draining states
graceful shutdown
```

Each worker advertises:

```text
worker and device ID
backend
free and total memory
loaded model revision
supported dtypes
rank and world size
communicator/group identity
basic topology information
```

The controller uses this information to form valid replicas or
tensor-parallel groups without accidentally assigning incompatible ranks or
oversubscribing one device.

---

## Lesson 77 — Distributed frontend and router

Lesson 73 already built the single-worker HTTP frontend. This lesson extends
that frontend with distributed placement instead of rebuilding the API.

The distributed frontend:

```text
validates request
assigns request ID
applies limits/deadlines
selects a replica group
forwards cancellation
streams tokens back
```

Begin with round-robin:

```text
request 1 → replica group A
request 2 → replica group B
request 3 → replica group A
```

Keep two schedulers conceptually separate:

```text
HTTP frontend
    validates and streams the external request

global placement/admission scheduler
    Which replica group receives the request?

worker-local scheduler
    Which sequences enter the next GPU batch?
```

---

## Lesson 78 — Replicated workers and admission control

The simplest multi-device strategy is replication with tensor-parallel size
one:

```text
replica group A → one worker with a complete model copy
replica group B → one worker with a complete model copy
replica group C → one worker with a complete model copy
```

Different requests run independently.

Add:

```text
per-group queue limits
group KV-memory capacity reports
global backpressure
overload rejection
draining
```

On one Mac, multiple worker processes remain architecturally real but contend
for the same integrated GPU and unified memory.

---

## Lesson 79 — Load-aware and KV-aware routing

Improve replica-group choice using aggregated group information:

```text
health of every required rank
group queue length
usable KV capacity across the group
loaded model revision
tokenizer/chat-template identity
adapter identity
dtype and cache-layout version
estimated prompt cost
request priority/deadline
```

After prefix caching exists, the router may ask:

```text
which replica group already owns the longest reusable prefix?
```

Routing to that group may avoid repeating prefill.

This introduces the trade-off:

```text
best cache locality
versus
best current load
```

---

## Lesson 80 — Tensor-parallel mathematics

Replication requires every device to fit the whole model.

Tensor parallelism splits one model operation across devices.

For:

```text
Y = X × W
```

These equations use the mathematical layout:

```text
W [K,N]
```

Hugging Face linear weights may still be physically stored as `[N,K]` and
read as transposed by our kernels.

Column-parallel split:

```text
W = [W0 | W1]

rank 0 computes Y0 = X × W0
rank 1 computes Y1 = X × W1

Y is logically [Y0 | Y1]
```

The output may stay sharded if the next operation can consume it that way.
Only a consumer requiring the complete tensor needs an all-gather or
concatenation.

Row-parallel split:

```text
X = [X0 | X1]

W =
[W0]
[W1]

rank 0 computes partial0 = X0 × W0
rank 1 computes partial1 = X1 × W1

Y = all_reduce_sum(partial0, partial1)
```

Begin with simulated ranks and CPU references.

---

## Lesson 81 — Collectives

Build miniature collective operations:

```text
broadcast
    one rank sends the same value to all ranks

all-gather
    ranks contribute different pieces
    every rank receives the concatenation

all-reduce
    ranks contribute partial results
    every rank receives the combined result

reduce-scatter
    partials are combined
    every rank receives one output section
```

First implement them with:

```text
threads
processes
shared memory or sockets
```

This teaches ordering, synchronization and partitioning correctness.

Efficient NVIDIA execution later uses NCCL.

---

## Lesson 82 — Tensor-parallel transformer execution

Apply Lessons 80 and 81 to:

```text
QKV projections
attention output projection
SwiGLU gate/up/down projections
vocabulary projection
```

Define:

```text
replica-group identity
rank and world size
communicator membership
rank-to-device placement
which weights are sharded
which activations are replicated
where all-gather or all-reduce occurs
how attention heads are divided
```

The router submits a request to the logical replica group. A coordinator
ensures all ranks execute compatible steps; no individual rank independently
owns the full request.

Correctness test:

```text
sharded logits == single-worker logits
```

within the selected precision tolerance.

On one Mac, this validates the architecture but not multi-GPU speed.

---

## Lesson 83 — Separate prefill and decode workers

Prefill and decode have different shapes:

```text
prefill
    many prompt rows
    large matrix operations
    compute-heavy

decode
    one row per active sequence
    repeated model/KV reads
    latency-sensitive
```

A disaggregated system can use:

```text
request
    ↓
prefill worker
    ↓ creates KV cache
KV-cache transfer
    ↓
decode worker
    ↓ streams tokens
```

Potential benefit:

```text
long prefills do not block decode batches
prefill/decode capacity can scale separately
workers can be tuned differently
```

Potential cost:

```text
moving the KV cache may be expensive
```

This is a trade-off, not an automatic speedup.

### First-token handoff contract

The miniature design will use one explicit protocol:

```text
prefill worker
    computes prompt logits
    samples and emits the first generated token
    keeps that token as an uncomputed pending decode input
        ↓
transfers:
    prompt KV cache
    prompt cache length
    pending first token ID
    sampling RNG state/counter
        ↓
decode worker
    consumes the pending token
    appends its K/V
    predicts the second generated token
```

If the first token is EOS, the request finishes without a decode-worker
handoff.

This removes ambiguity about whether the transferred cache already contains
the first generated token.

---

## Lesson 84 — Transactional KV-cache transfer

The decode worker needs K/V for every layer, not merely the original prompt.

Transfer metadata includes:

```text
transfer ID
request ID
model/cache format identity
token count
logical token ranges and valid counts
source-side layout metadata
tensor shapes
dtypes
K/V bytes
checksums or validation data
```

Physical block IDs are local to one worker's pool. The destination does not
adopt the source's numeric block IDs. It allocates its own blocks and builds
a new destination-local block table while preserving logical token order.

Safe ownership protocol:

```text
1. Create a unique transfer_id.
2. Destination enters PREPARED and reserves local blocks.
3. Source sends metadata and payloads in logical token order.
4. Destination validates completeness and checksums.
5. Destination atomically enters COMMITTED.
6. Destination acknowledges the committed transfer_id.
7. Router changes request ownership only after confirmed commitment.
8. Source enters RELEASED and frees its blocks when safe.
```

Operations are idempotent:

```text
PREPARED
COMMITTED
RELEASED
```

If an acknowledgement is lost, the controller queries transfer status using
`transfer_id`; it does not start an unrelated second transfer or release the
source blindly.

If transfer fails before commitment:

```text
destination discards incomplete blocks
source retains ownership
request retries or stays local
```

Transport progression:

```text
in-process copy
    ↓
shared memory
    ↓
TCP
    ↓
GPU-aware transfer on suitable hardware
```

---

## Lesson 85 — Failure handling and retries

Distributed systems have partial failures:

```text
worker dies during prefill
worker dies during decode
KV transfer stops halfway
request completed but acknowledgement was lost
worker becomes slow without crashing
```

State needs identities such as:

```text
request_id
generation_epoch
decode_step
cache_version
worker_id
```

These reject stale or duplicated operations.

### Recovery examples

Prefill failure:

```text
retry the prompt on another healthy worker
```

Decode-worker failure:

```text
replay prompt + generated token history
rebuild KV cache
restore the per-request sampling seed/counter or RNG state
continue
```

Token history alone does not guarantee the same stochastic continuation.
Sampling state must be recoverable or derived deterministically from
`request_id` and decode step.

Duplicate decode message:

```text
return the previously committed result
instead of appending the same token twice
```

The worker therefore retains the committed token/result identity for every
step that may be retried.

---

## Lesson 86 — Distributed metrics and tracing

Lessons 74–85 retain minimal structured logs, request IDs and counters so
they are debuggable. This lesson combines them into a coherent distributed
metrics and tracing system.

Trace the complete request:

```text
HTTP handling
tokenization
router queue
worker queue
prefill
KV transfer
decode
streaming
```

System metrics:

```text
TTFT
TPOT and inter-token latency
input-token throughput
output-token throughput
prefill throughput
decode throughput
router and worker queue latency
p50, p95 and p99 request latency
model, KV-cache and temporary memory
device utilization
worker health
active/queued requests
free KV blocks
prefix-cache hit rate
batch sizes
transfer bytes and latency
RPC failures/timeouts
retries
worker restarts
```

One request trace might look like:

```text
request 42
├── router queue
├── worker queue
├── prefill
├── KV transfer
└── decode steps
```

---

## Lesson 87 — Ordinary multi-host networking

Move the already-correct worker and router processes onto more than one
ordinary machine.

This does not require NVIDIA hardware.

Build and test:

```text
network addresses and service discovery
cross-host TCP RPC
authentication and encrypted transport where appropriate
timeouts and reconnects
host-level placement
clock-independent tracing
bandwidth and latency measurements
multi-host failure injection
```

At this stage, model execution may still use CPU mocks, Metal on a reachable
Mac, or another available backend.

---

## Lesson 88 — NVIDIA specialization

Add an NVIDIA backend while preserving the higher-level contracts.

Replace or add backends:

```text
Metal queues/events       → CUDA streams/events
Metal buffers             → CUDA allocations
software collectives      → NCCL collectives
ordinary network copies   → GPU-aware networking where available
```

Measure:

```text
scaling efficiency
collective communication time
KV-transfer cost
interconnect bandwidth
worker imbalance
```

This is where CUDA-, Tensor-Core-, NCCL- and NVLink-specific learning becomes
unavoidable.

RDMA requires suitable server networking and software support; it is not
conceptually NVIDIA-only. GPU-direct RDMA experiments additionally require
compatible accelerators, NICs and platform support.

---

## Lesson 89 — Complete Mini-Dynamo

Final miniature system:

```text
streaming frontend
        ↓
global router and admission control
        ↓
healthy model workers
        ↓
local Mini-vLLM engines
        ↓
optional replicas or tensor-parallel groups
        ↓
optional separate prefill/decode pools
        ↓
KV-cache transfer
        ↓
metrics, cancellation and failure recovery
```

Final correctness rule:

```text
distributed generated token IDs
        ==
single-worker baseline token IDs
```

when using equivalent deterministic settings.

Final acceptance also requires:

```text
no duplicated or skipped output tokens
no leaked KV blocks
correct cancellation
working overload backpressure
idempotent retry behavior
successful recovery from tested worker failures
consistent request traces and metrics
```

---

# What can be built on the Mac?

## Fully real on the current Mac

```text
request lifecycle
continuous batching
paged KV allocation
paged attention
memory-aware scheduling
buffer pools
asynchronous Metal submission
FP16 inference
BF16 where the selected Metal hardware/software path supports it
GQA/MQA
optimized Metal kernels
kernel fusion
online softmax and FlashAttention principles
quantized storage and kernels
prefix caching
speculative decoding
profiling
streaming HTTP service
multiple processes
RPC, routing, health and retries
KV transfer through ordinary memory or sockets
```

## Architecturally real but performance-simulated on one Mac

```text
multiple model workers sharing one physical GPU
logical multi-device scheduling
tensor-parallel ranks
software collectives
separate prefill/decode processes
multi-worker scaling experiments
```

These teach ownership, partitioning, protocols and correctness, but one
integrated GPU cannot demonstrate genuine multi-GPU scaling.

## Requires NVIDIA or server hardware for the real backend/performance work

```text
CUDA kernel implementation and tuning
Tensor Core-specific optimization
NCCL collectives
NVLink behavior
PCIe multi-GPU bottlenecks
real multi-node GPU scaling
GPUDirect on compatible NVIDIA/server platforms
RDMA experiments on suitable server NICs and networking
large-scale deployment and failure characteristics
```

The Mac remains useful until the question changes from:

```text
Do I understand and correctly implement this architecture?
```

to:

```text
How efficiently does this run on real NVIDIA clusters?
```

---

# Milestones

```text
Lesson 39
    Correct cached single-request Llama.

Lesson 49
    Correct Mini-vLLM with paged KV cache and continuous batching.

Lesson 55
    A meaningful trained, sharded, mixed-precision model runs.

Lesson 63
    Core matmul, fusion and attention paths are substantially optimized.

Lesson 68
    The compiler participates in transformer planning and fusion.

Lesson 73
    One optimized worker serves streaming HTTP requests.

Lesson 89
    Mini-Dynamo coordinates multiple workers and distributed execution.
```

## Dependency summary

```text
Lesson 39 cached decoding
        ↓
Lessons 40–49 Mini-vLLM correctness
        ↓
Lesson 50 trusted measurements
        ↓
Lessons 51–65 runtime, precision and kernel optimization
        ↓
Lessons 66–68 compiler integration
        ↓
Lessons 69–73 advanced serving features and HTTP
        ↓
Lessons 74–89 Mini-Dynamo
        ↓
real CUDA/NVIDIA specialization
```

## Resume point

When development resumes, do not jump directly to paged attention or
continuous batching.

Begin with:

```text
Lesson 40 — Requests, sequences and lifecycle
```

That lesson creates the state objects whose ownership will later connect:

```text
request
    ↓
sequence
    ↓
block table
    ↓
KV blocks
    ↓
scheduler
    ↓
batched model execution
```

