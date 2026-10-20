# Inference Lab — Future Diffusion Roadmap

## Status

This document is a future plan.

No diffusion executor has been implemented in Infer Lab yet. The current
completed checkpoint remains Lesson 39:

```text
small Hugging Face Llama
        ↓
prompt prefill
        ↓
per-layer KV cache
        ↓
cached token-by-token decoding
```

The existing [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) continues the language
model path toward Mini-vLLM and Mini-Dynamo.

This document describes a parallel diffusion path. It uses lesson identifiers
such as `D1`, `D2`, and `D3` so it does not collide with Lessons 40–89.

Here, “diffusion” primarily means image and latent diffusion or flow models.
Discrete diffusion language models would require a separate model-specific
execution path.

The roadmap should still be implemented one lesson at a time. Every lesson
should begin with a simple explanation, one small numerical example, and a
clear correctness test before optimization is added.

---

## Final goal

The long-term goal is a shared miniature inference server that can execute
both language and diffusion models:

```text
requests
   │
   ▼
MiniServe
   │
   ├── LLM executor
   │      prompt → prefill → token decode → text
   │
   └── Diffusion executor
          prompt → denoising steps → VAE decode → image
```

The diffusion path should eventually support:

```text
text-to-image requests
deterministic seeds
multiple schedulers
negative prompts
classifier-free guidance
tiny DiT and U-Net models
real local safetensors checkpoints
multiple concurrent requests
step-aware batching
memory admission and cancellation
prompt-embedding caching
FP16 inference
profiling
streaming previews
HTTP serving
multi-worker execution
```

The first target is not a production Stable Diffusion or FLUX server.

The first success criterion is:

```text
one tiny deterministic diffusion pipeline
running through MiniTensor and MiniGPU
matching a Python reference at important boundaries
```

---

## The central architectural decision

We should not turn the Llama executor into a generic class containing many
diffusion-specific conditions.

Instead, shared serving mechanics should live in a small `MiniServe` layer,
while each model family keeps its own typed scheduler, request state, and
executor.

```text
HTTP or in-process API
          │
          ▼
┌────────────────────────────────────────┐
│               MiniServe                │
│                                        │
│ request IDs         output events      │
│ admission           cancellation       │
│ priorities          metrics            │
│ worker loops        error propagation  │
└────────────────────┬───────────────────┘
                     │
              model-kind router
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐   ┌────────────────────┐
│   LLM domain     │   │ Diffusion domain   │
│                  │   │                    │
│ LLMScheduler     │   │ DiffusionRequest-  │
│                  │   │ Scheduler          │
│ SequenceState    │   │ SampleState        │
│ paged KV manager │   │ latent manager     │
│ LLMExecutor      │   │ DiffusionExecutor  │
└─────────┬────────┘   └─────────┬──────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          MiniTensor and compiler
                     ▼
             MiniGPU / Metal
                     ▼
                 Apple GPU
```

`MiniServe` should share mechanics, not pretend that language generation and
image generation have identical state.

An LLM request owns a growing token sequence and KV cache.

A diffusion request owns a latent image tensor that is repeatedly updated.

---

## What diffusion inference does

An autoregressive language model repeatedly predicts one new token:

```text
prompt tokens
     ↓
predict next-token logits
     ↓
choose one token
     ↓
append it
     ↓
repeat
```

A diffusion pipeline starts with random noise and repeatedly removes predicted
noise or follows a learned flow:

```text
prompt
  ↓
text embeddings
  ↓
initial random latent xT
  ↓
denoiser at step T
  ↓
scheduler update
  ↓
denoiser at step T-1
  ↓
scheduler update
  ↓
...
  ↓
clean latent x0
  ↓
VAE decoder
  ↓
RGB image
```

For a simple diffusion schedule, forward noising can be written as:

```text
x_t =
    sqrt(alpha_bar_t) * x_0
  + sqrt(1 - alpha_bar_t) * noise
```

The neural network receives:

```text
current noisy latent
current timestep or sigma
prompt conditioning
```

and predicts information used by a scheduler to produce the next latent.

Different model families may predict:

```text
noise
clean samples
velocity
flow
```

The executor should therefore avoid hard-coding one prediction meaning into
the general request scheduler.

---

## Three terms that must remain separate

### Noise scheduler or sampler

This is mathematical model code. It uses a denoiser prediction to update:

```text
current latent → next latent
```

Examples include DDIM-, Euler-, or DDPM-style updates.

### Request scheduler

This is serving code. It decides:

```text
which requests run next
which requests can share a batch
how much memory and compute may be admitted
```

### VAE decode

This converts a final latent into image pixels.

It is unrelated to LLM token decoding, even though both use the word
“decode.”

---

## LLM and diffusion comparison

| Concept | LLM generation | Diffusion generation |
| --- | --- | --- |
| Persistent request data | Token IDs and per-layer KV cache | Current latent, prompt embeddings, step state |
| Repeated work unit | One decode token | One denoising step |
| State growth | KV cache grows with tokens | Latent usually keeps a fixed shape |
| Main progress counter | Token position | Timestep, sigma, or step index |
| Natural output stream | Tokens | Usually final image; previews are optional |
| Main batching key | Model, phase, cache/token constraints | Model, stage, shape, dtype, CFG, scheduler |
| Main memory concern | Persistent KV blocks | Large temporary activations and VAE workspace |
| Common reusable cache | Prefix/KV cache | Prompt embeddings and compiled pipelines |
| Sampling randomness | Choose among vocabulary logits | Generate initial or per-step noise |

Both workloads need:

```text
request lifecycle
admission control
batch construction
device workers
asynchronous completion
temporary-buffer reuse
cancellation
metrics
model loading
```

Their mathematical work units and persistent states remain different.

---

## What the current project can reuse

### Directly reusable

The following foundations already exist or are planned for the LLM path:

- `minigpu::Device`, `Queue`, `Kernel`, `Dispatch`, `Buffer`, and `Event`.
- Real Apple Metal kernel execution.
- Launch geometry and padded-thread guards.
- Shared and private Metal storage concepts.
- Threadgroup memory, barriers, reductions, and atomics.
- `RuntimeTensor`, shapes, strides, views, and contiguous materialization.
- Tensor storage ownership and buffer lifetime management.
- Elementwise code generation and compiler fusion ideas.
- Stable softmax and reductions.
- Linear layers, matmul, tiled matmul, and batched matmul.
- Embeddings, SiLU, attention, normalization, and residual connections.
- Safetensors and JSON parsing patterns.
- CPU/Python reference comparisons.
- Future asynchronous execution and event-aware memory pools.
- Future dtype, quantization, profiling, worker, RPC, and HTTP layers.

### Reusable after generalization

Attention is reusable, but current Llama attention is causal:

```text
token cannot attend to future token
```

Diffusion transformers normally need:

```text
bidirectional latent self-attention
cross-attention from latent tokens to prompt embeddings
```

The attention implementation should therefore separate:

```text
Q/K/V projections
score calculation
mask policy
softmax
value collection
```

from Llama-specific causal masking.

Current RMSNorm, RoPE, and SwiGLU may be useful for some architectures, but
diffusion models can also require:

```text
LayerNorm
GroupNorm
AdaLN or other adaptive normalization
GELU
timestep conditioning
2D position embeddings
```

The Hugging Face loader structure is reusable, but diffusion checkpoints have
different components and tensor-name mappings.

---

## What must be added

### Core diffusion mathematics

- Forward noising.
- Timestep, alpha, sigma, and coefficient schedules.
- A scheduler interface.
- Deterministic and stochastic scheduler steps.
- Support for different model prediction types.

### Random-number generation

- Per-request seeds.
- Normal-distribution sampling.
- Exact test fixtures for initial latents.
- A deterministic counter-based GPU RNG later.
- RNG behavior independent of batch order and GPU lane scheduling.

### Image and latent tensors

- Four-dimensional tensors such as `[B, C, H, W]`.
- Clear channel and pixel layouts.
- Latent scaling and image normalization.
- Patchify and unpatchify for diffusion transformers.
- CPU image conversion and PNG writing.

### Diffusion Transformer path

- Non-causal self-attention.
- Cross-attention to prompt embeddings.
- Timestep embeddings.
- LayerNorm or adaptive LayerNorm.
- Position embeddings for image patches.
- A small DiT denoiser.

This should be the first neural denoiser path because it reuses more of the
existing transformer and matmul work.

### U-Net path

- Conv2D.
- GroupNorm.
- Residual convolution blocks.
- Downsampling and upsampling.
- Skip connections.
- Attention inside selected spatial blocks.

This path is important for Stable-Diffusion-style U-Nets, but it requires more
new spatial operators than a DiT-first path.

### Text conditioning

- Tokenizer integration.
- A CLIP-like or T5-like text encoder interface.
- Prompt embeddings.
- Negative-prompt embeddings.
- Cross-attention.
- Prompt-embedding caching.

### Classifier-free guidance

The first implementation can run two predictions:

```text
unconditional_prediction
conditional_prediction
```

and combine them:

```text
guided =
    unconditional
  + guidance_scale
    * (conditional - unconditional)
```

Later, both predictions can be placed in one `2B` GPU batch.

### VAE

- Latent-to-image decoding.
- Correct latent scaling constants.
- Convolutions, normalization, activations, and upsampling.
- RGB range conversion.
- Optional VAE encoding later for image-to-image workloads.

---

## Recommended model strategy

We should not begin with a large production checkpoint.

Use progressively larger targets:

```text
hand-written numeric example
        ↓
tiny random CPU denoiser
        ↓
tiny random Metal denoiser
        ↓
small unconditional image model
        ↓
tiny text-conditioned pipeline
        ↓
pinned Hugging Face test checkpoint
        ↓
small meaningful trained model
```

The first committed neural fixture should be small enough that its generated
weights can live in the repository.

Downloaded third-party checkpoints should remain excluded from Git, following
the existing Llama model-fixture convention.

Every external model must pin:

```text
repository
exact revision
model configuration
scheduler configuration
expected file layout
license notes
```

Modern image generators may use traditional diffusion, rectified flow, or
flow matching. The high-level executor should expose a generic denoising or
flow-step interface rather than assuming every model uses the same DDPM
equations.

---

## Proposed public request

A future request might contain:

```cpp
struct DiffusionRequest {
    std::string prompt;
    std::string negative_prompt;
    std::string model_id;

    std::uint64_t seed;
    std::size_t width;
    std::size_t height;
    std::size_t step_count;
    std::size_t image_count;

    float guidance_scale;
    NoiseSchedulerKind scheduler;
    DiffusionOutputFormat output_format;
};
```

This is only a conceptual API. Exact types should be chosen when the lesson is
implemented.

The executor-owned state will be closer to:

```text
request ID
prompt and negative-prompt embeddings
current latent tensor
current step index
timestep or sigma sequence
scheduler history
RNG key and counter
workspace handles
latest completion event
final image or error
```

This state must not be forced into the LLM `SequenceState` type.

---

## Diffusion request lifecycle

```text
Waiting
   ↓
Validating
   ↓
TextEncoding
   ↓
LatentInitialization
   ↓
Denoising step 0
   ↓
Denoising step 1
   ↓
...
   ↓
Denoising step N - 1
   ↓
VAEDecoding
   ↓
ImageEncoding
   ↓
Finished
```

A request may also become:

```text
Cancelled
Failed
```

at any boundary.

Cancellation stops future scheduling immediately. Buffers referenced by an
already submitted GPU command remain alive until its completion event.

The request step must advance transactionally:

```text
submit complete denoiser work
        ↓
submit scheduler update
        ↓
wait for or observe successful completion
        ↓
commit current_step = current_step + 1
```

If the GPU work fails, the engine must not pretend that the request reached the
next step.

---

## Scheduling and continuous batching

### Why LLM batching cannot be copied directly

Mini-vLLM will schedule prompt tokens, decode tokens, and KV-cache blocks.

A diffusion scheduler instead schedules pipeline stages:

```text
text encoder batch
denoiser-step batch
VAE decode batch
image encode work
```

The first teaching scheduler should batch only requests with the same:

```text
model or pipeline
execution stage
latent height and width
dtype
classifier-free-guidance mode
scheduler kind
current step or timestep
```

This creates an explicit compatibility key:

```cpp
struct DiffusionCompatibilityKey {
    ModelId model;
    DiffusionStage stage;
    Shape latent_shape;
    DType dtype;
    bool classifier_free_guidance;
    NoiseSchedulerKind scheduler;
    std::size_t step_index;
};
```

Later, the scheduler may batch different timesteps by supplying a different
timestep embedding for every batch row.

An engine iteration can look like:

```text
accept and cancel requests
        ↓
collect compatible ready work
        ↓
build a text, denoiser, or VAE batch
        ↓
submit GPU work asynchronously
        ↓
map batch rows back to request IDs
        ↓
advance successful requests
        ↓
emit progress, image, or error events
        ↓
reclaim completed resources
```

Fairness matters. A high-resolution request with many steps must not
permanently block shorter jobs.

The scheduler should eventually budget something closer to:

```text
latent pixels
× model cost
× batch size
× remaining steps
```

rather than an LLM token budget.

---

## Memory model

### LLM request memory

```text
token sequence
per-layer K/V rows
KV memory grows as tokens are added
```

### Diffusion request memory

```text
current latent
prompt embeddings
timestep and scheduler state
possibly a few previous predictions for a multi-step solver
```

Diffusion normally does not retain every previous latent or every intermediate
activation.

The large denoiser and VAE activations are temporary and should be returned to
an event-aware workspace pool after GPU completion.

Admission control must estimate:

```text
shared model weights
+ persistent request latent
+ prompt and negative-prompt embeddings
+ scheduler history
+ denoiser workspace for the proposed batch
+ VAE decode workspace
+ safety margin
```

Resolution is critical. Doubling both image dimensions approximately
quadruples the number of image or latent positions.

---

## Diffusion caching

Paged token KV caching is not the general diffusion cache.

Useful diffusion caches include:

### Prompt-embedding cache

Keyed by all inputs that affect text encoding:

```text
text-encoder model revision
token IDs
attention mask
negative prompt
dtype
encoder configuration
```

### Compiled-kernel and execution-plan cache

Keyed by:

```text
operator or fused graph
model revision
shape
dtype
device capability
```

### Constant scheduler tables

Precompute timestep, alpha, sigma, and solver coefficient tables.

### Optional cross-attention projection cache

Prompt embeddings stay constant while the latent changes. Some pipelines may
benefit from caching text-side K/V projections.

### Advanced step caching

Later experiments may reuse or approximate denoiser features across neighboring
steps. This must remain an optional optimization with explicit quality and
correctness measurements.

These caches live in CPU or GPU-accessible memory unless a later feature
explicitly introduces disk offload. They should not be described as storing
active request state on disk.

---

## Suggested diffusion lessons

## Stage D1 — Fundamentals on the CPU

### D1 — What diffusion inference is

Build no neural network yet.

Use a tiny vector to explain:

```text
clean value
add known noise
ask an ideal denoiser for that noise
apply one reverse update
```

The lesson establishes the vocabulary:

```text
sample
noise
timestep
schedule
denoiser
scheduler
latent
```

### D2 — Forward noising

Implement:

```text
beta schedule
alpha schedule
cumulative alpha
q_sample(x0, noise, timestep)
```

Compare every number with a simple Python or NumPy reference.

### D3 — Deterministic random noise

Begin with host-generated, saved FP32 latent fixtures.

Later add a counter-based RNG conceptually keyed by:

```text
seed
request ID
sample index
diffusion step
tensor element index
```

The same request must receive the same noise regardless of unrelated batch
membership or GPU scheduling order.

### D4 — Noise-scheduler interface

Implement one deterministic scheduler first.

The interface should separate:

```text
model prediction
current latent
current timestep
scheduler coefficients
next latent
```

Stochastic schedulers come only after deterministic parity passes.

### D5 — Latent tensors and image layout

Introduce:

```text
[batch, channels, height, width]
```

Verify flattening, strides, channel order, padding, and awkward image sizes.

Render a tiny tensor as a simple grayscale image on the CPU so the data layout
becomes visible.

---

## Stage D2 — First Metal diffusion loop

### D6 — Latent elementwise Metal kernels

Implement the operations needed by a scheduler update:

```text
scale
add
subtract
multiply
linear combination
clamp
```

The compiler may fuse these into one kernel later.

### D7 — Timestep and sigma embeddings

Build sinusoidal timestep embeddings, followed by a small projection.

Explain how every image position receives information about the current
denoising step.

### D8 — Missing normalization and activation operations

Add the minimal operators needed by the chosen denoiser:

```text
LayerNorm or GroupNorm
GELU where needed
adaptive scale and shift
```

### D9 — Non-causal and cross-attention

Generalize current attention:

```text
latent self-attention:
    Q, K, and V all come from latent tokens

cross-attention:
    Q comes from latent tokens
    K and V come from prompt embeddings
```

No causal future-token mask is used for ordinary image-patch attention.

### D10 — Tiny Diffusion Transformer block

Build one small DiT block using:

```text
patch tokens
time conditioning
non-causal attention
MLP
residual connections
```

Compare every intermediate activation with a Python reference.

### D11 — Tiny unconditional denoiser

Stack a few tiny DiT blocks and predict noise for a very small latent.

No tokenizer, prompt, CFG, or VAE yet.

### D12 — Complete unconditional denoising loop

Run:

```text
initial latent
  → denoiser
  → scheduler
  → denoiser
  → scheduler
  → final small image tensor
```

This is the first complete GPU diffusion pipeline.

---

## Stage D3 — Text conditioning and guidance

### D13 — Prompt-embedding fixtures

Initially generate small prompt embeddings in Python and load them from a
fixture.

This separates cross-attention correctness from tokenizer and text-encoder
correctness.

### D14 — Conditional denoiser

Feed prompt embeddings through cross-attention and prove that different
conditioning tensors change the prediction.

### D15 — Negative prompts and classifier-free guidance

Implement conditional and unconditional predictions separately.

Then add the guidance-combination kernel.

### D16 — Packed CFG batch

Pack:

```text
unconditional rows
conditional rows
```

into one `2B` denoiser batch, then split and combine the predictions.

### D17 — Text tokenizer and encoder integration

Add a small text-encoder interface.

The first native diffusion milestone does not need to implement a tokenizer
from scratch. A Python bridge, like Lesson 36, is acceptable before a native
tokenizer is built.

---

## Stage D4 — VAE and a real latent pipeline

### D18 — VAE building blocks

Implement only the operations required by a tiny decoder:

```text
Conv2D
normalization
SiLU
residual blocks
upsampling
```

### D19 — Tiny VAE decoder

Decode a known latent fixture and compare the output image tensor with Python.

Verify:

```text
latent scaling
channel order
output range
RGB conversion
```

### D20 — Image encoding

Convert the final tensor to a PNG on the CPU.

GPU-native PNG encoding is unnecessary for the first engine.

### D21 — Full tiny text-to-image pipeline

Connect:

```text
prompt embeddings
        ↓
initial latent
        ↓
conditional denoising loop
        ↓
VAE decoder
        ↓
PNG image
```

This is the first end-to-end diffusion milestone.

---

## Stage D5 — U-Net and model-family support

### D22 — Conv2D optimization and tests

Start with a direct convolution kernel, then add tiled or im2col experiments.

Correctness across stride, padding, channels, and odd sizes comes first.

### D23 — GroupNorm and residual convolution blocks

Build the core block used by many latent-diffusion U-Nets.

### D24 — Downsampling, upsampling, and skip connections

Track shapes carefully through:

```text
encoder path
bottleneck
decoder path
```

### D25 — Tiny U-Net denoiser

Run a small U-Net through the same scheduler abstraction already used by the
DiT path.

### D26 — Generic denoiser boundary

Both models should satisfy one conceptual interface:

```cpp
Tensor predict(
    Tensor latent,
    Tensor timestep_or_sigma,
    Conditioning conditioning
);
```

The scheduler should not care whether the prediction came from a DiT, U-Net,
or later flow transformer.

---

## Stage D6 — Real checkpoint integration

### D27 — Diffusion config and safetensors loader

Generalize model loading for component trees:

```text
text encoder
denoiser
VAE
scheduler configuration
tokenizer files
```

### D28 — Pinned tiny external checkpoint

Select one deliberately tiny checkpoint and pin its exact revision.

Document the license, expected files, dtype, architecture, and limitations.

### D29 — Per-component parity

Compare independently:

```text
text embeddings
one denoiser call
one scheduler update
VAE output
```

### D30 — End-to-end parity

Use the same:

```text
prompt
negative prompt
initial latent bytes
timestep sequence
scheduler configuration
guidance scale
```

in both the Python reference and the Metal engine.

Compare intermediate latents instead of judging only whether the final image
looks similar.

---

## Stage D7 — Diffusion serving

### D31 — `DiffusionRequest` and `SampleState`

Add typed request validation and lifecycle transitions.

### D32 — Single-request `DiffusionExecutor`

Move the working pipeline behind an executor boundary without adding batching
yet.

### D33 — Memory estimates and admission control

Estimate persistent latent memory and peak temporary workspace before accepting
a request.

### D34 — Step-aware request scheduler

Create compatibility buckets and build one denoiser-step batch at a time.

### D35 — Dynamic joining and leaving

Allow requests to enter and finish while other requests continue through their
denoising loops.

### D36 — Cancellation and failure recovery

Cancelled requests stop receiving future steps. In-flight buffers remain alive
until GPU completion.

### D37 — Prompt-embedding and plan caches

Add bounded caches, cache keys, eviction, and memory accounting.

### D38 — Progress and preview events

Emit:

```text
accepted
text encoded
step completed
preview available
final image
cancelled
failed
```

Preview decoding should be optional because it adds VAE work.

### D39 — Streaming HTTP API

Expose request creation, status, cancellation, metadata, preview events, and
final image retrieval.

At this point the shared layer can be called `MiniServe` or `Mini-Omni`.

---

## Stage D8 — Optimization

### D40 — Asynchronous execution

Stop waiting after every kernel. Chain work through queue ordering and events.

### D41 — Event-aware workspace pools

Reuse denoiser and VAE temporary buffers only after their final GPU event.

### D42 — FP16 inference

Store large weights and activations in FP16 while keeping sensitive reductions
and accumulations in FP32 where required.

FP16 should precede BF16 because BF16 availability and performance are more
device-dependent.

### D43 — Fused diffusion kernels

Candidate fusions include:

```text
normalization + projection
bias + activation
CFG combination
scheduler update
QKV projection
residual + adaptive scale/shift
```

### D44 — Optimized attention

Apply tiled or online-softmax attention to DiT and U-Net attention blocks.

### D45 — Optimized convolution

Compare direct, tiled, im2col-plus-matmul, and specialized small-kernel
strategies.

### D46 — Quantized weights

Begin with weight-only quantization for linear and convolution layers.

Image quality must be evaluated as well as tensor error and speed.

### D47 — Advanced diffusion caching

Experiment with cross-attention projection reuse and step-feature caching.

Every approximation must report:

```text
latency saved
memory used
numerical difference
image-quality effect
```

### D48 — Profiling and performance tuning

Measure:

```text
cold model load
cold Metal compilation
queue time
text-encoder time
average denoiser-step time
steps per second
denoiser batch utilization
VAE decode time
PNG encoding time
end-to-end latency
images per minute
peak persistent memory
peak temporary memory
prompt-cache hit rate
```

Use warm-up runs, medians, and p90 measurements. State exactly what every
timer includes.

---

## Stage D9 — Multi-worker and Mini-Dynamo integration

### D49 — Diffusion worker protocol

Define RPC messages for:

```text
load pipeline
submit request
advance or schedule work
cancel request
emit progress
return image
report health and memory
```

### D50 — Stage-specialized workers

Experiment with separate workers for:

```text
text encoding
denoising
VAE decoding
image encoding
```

This resembles separating LLM prefill and decode workers, but the transfer
objects are prompt embeddings and latents rather than KV-cache blocks.

### D51 — Multi-device execution

Explore:

```text
model replication
request routing
pipeline parallelism
tensor parallelism
component placement
```

### D52 — Unified Mini-Dynamo router

The distributed frontend routes each request to the correct serving domain:

```text
text-generation request
        ↓
LLM workers

image-generation request
        ↓
diffusion workers
```

Shared health management, metrics, retries, and worker discovery remain in the
Mini-Dynamo control plane.

---

## Correctness strategy

Diffusion output is random, and small numerical differences can accumulate
over many steps.

We should not begin with only:

```text
Does the final image look similar?
```

Verification should proceed hierarchically:

```text
individual operator
        ↓
one block
        ↓
one denoiser call
        ↓
one scheduler update
        ↓
several denoising steps
        ↓
VAE-decoded image
```

Use two types of reference:

```text
simple Python or NumPy
    formulas, layouts, schedules, RNG fixtures

PyTorch and Hugging Face tooling
    checkpoint loading, neural blocks, complete pipeline
```

Every reproducible fixture should record:

```text
model repository and revision
scheduler configuration
seed
initial latent
prompt or conditioning
timestep sequence
expected intermediate tensors
software versions
```

### Exact latent-fixture mode

Generate the initial FP32 latent once in Python and save its exact bytes.

Both Python and Metal consume the same file. This removes RNG differences from
model and scheduler parity tests.

### Engine RNG mode

Test the raw counter-based RNG output separately.

The same request seed and element indexes must produce the same sequence
regardless of:

```text
threadgroup size
GPU lane scheduling
batch order
unrelated requests
```

### Measurements per denoising step

Record:

```text
maximum absolute error
mean absolute error
relative error
presence of NaN or infinity
```

Tolerances should be chosen per operation and dtype. They must not be silently
widened just to make a failing test pass.

### Batch invariance

A deterministic request should produce the same result when run:

```text
alone
inside a compatible batch
beside a cancelled request
after another request completes
with a different physical batch-row position
```

This checks request isolation, row mapping, RNG independence, and workspace
reuse.

---

## Core invariants

1. One request cannot read another request's latent or conditioning except
   through an explicitly constructed batch.
2. Batch rows map explicitly back to request and sample IDs.
3. A request advances its step only after the complete update succeeds.
4. RNG results are deterministic per request and independent of batch order.
5. Temporary buffers are reused only after GPU completion.
6. Cancellation prevents new work but does not destroy in-flight storage.
7. Prompt-cache keys include every input that can change the embedding.
8. Scheduler coefficients correspond exactly to the selected timestep.
9. Model prediction type and scheduler interpretation must agree.
10. Output dimensions and memory requirements are validated before expensive
    GPU work begins.
11. Failed work does not corrupt unrelated requests.
12. Optimizations preserve the unfused reference result within documented
    tolerances.

---

## Mac and Metal scope

The existing project was developed on an Apple M4 Pro, but the diffusion path
should target Apple Silicon and Metal generally rather than hard-coding M4
assumptions.

The Mac is suitable for:

- Learning the complete architecture.
- Implementing diffusion mathematics.
- Running tiny DiT and U-Net models.
- Testing real Metal kernels.
- Building step-aware schedulers and memory pools.
- Serving small local checkpoints.

Important limitations:

- The current runtime is primarily FP32.
- Current operators often synchronize after each kernel.
- Many temporary tensors are allocated eagerly.
- Large image models have substantial weight and activation memory.
- Unified memory is shared with macOS and other applications.
- A model fitting in memory does not mean the naive engine will be fast.
- Metal threadgroup and matrix capabilities differ across Apple GPUs.

Practical diffusion checkpoints make these high priorities:

```text
FP16
asynchronous command submission
temporary-buffer reuse
optimized matmul and convolution
fused normalization and activation
optimized attention
```

CUDA, tensor cores, NCCL, NVLink, RDMA, and NVIDIA multi-GPU tuning remain a
later hardware-specific branch.

---

## Features deliberately deferred

Do not place these in the first diffusion milestone:

- SDXL- or FLUX-scale production checkpoints.
- High-resolution generation.
- Many scheduler families.
- BF16 and quantization.
- FlashAttention.
- LoRA.
- ControlNet.
- IP-Adapter.
- Inpainting.
- Image-to-image.
- Video diffusion.
- Disk offload.
- Multi-GPU sharding.
- GPU-native tokenization.
- Production image-quality evaluation.
- Distributed Mini-Dynamo deployment.

These are valuable only after one tiny pipeline is completely understood and
verified.

---

## Relationship to vLLM-style serving

The reusable idea from vLLM is not specifically the token KV cache.

The reusable idea is:

```text
separate model mathematics
from
request scheduling, memory management, batching, and serving
```

For language models:

```text
model work unit = prefill or decode token
persistent state = paged KV cache
```

For diffusion:

```text
model work unit = encoder, denoiser step, or VAE decode
persistent state = latent and conditioning
```

The official vLLM-Omni project demonstrates the same broad architectural
direction with diffusion-specific execution inside a larger serving system:

- [Adding a diffusion model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model/)
- [Diffusion features](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_features/)
- [vLLM-Omni documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)

Our educational implementation should remain smaller and reveal each layer
explicitly.

---

## Final architecture

```text
streaming text and image API
              ↓
Mini-Dynamo routing and worker management
              ↓
MiniServe request lifecycle and metrics
          ┌───┴────────────────┐
          ↓                    ↓
 Mini-vLLM domain       Mini-Diffusion domain
 token scheduler        step scheduler
 paged KV manager       latent/workspace manager
 LLM executor           diffusion executor
          └──────────┬─────────┘
                     ↓
       transformer and spatial operators
                     ↓
        MiniTensor and compiler
                     ↓
         MiniGPU / Metal runtime
                     ↓
                 Apple GPU
```

At that point, Infer Lab will demonstrate two major inference workloads:

```text
autoregressive sequence generation
iterative image generation
```

using one shared low-level runtime and one shared serving control plane, while
preserving the model-specific execution rules of each domain.
