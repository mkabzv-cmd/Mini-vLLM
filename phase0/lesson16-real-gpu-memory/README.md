# Lesson 16 — Real GPU memory

## Outcome

This lesson answers one question:

> Where does tensor data live, who can access it, and when is that access
> safe?

The same Metal kernel runs through two memory paths:

```text
Path A: Shared storage

CPU writes shared input
        ↓
GPU computes into shared output
        ↓
CPU waits for completion
        ↓
CPU reads shared output
```

```text
Path B: Private storage

CPU writes shared upload staging
        ↓ GPU blit copy
private GPU input
        ↓ GPU kernel
private GPU output
        ↓ GPU blit copy
shared readback staging
        ↓ wait for completion
CPU reads result
```

Both paths calculate:

```text
output = input × 3 + 1

input:  [1, 2, 3, 4, 5, 6, 7, 8]
output: [4, 7, 10, 13, 16, 19, 22, 25]
```

Only two files were added:

```text
README.md
demo.mm
```

Lesson 14 and the later lessons were not changed.

## Start with three separate questions

These ideas are related but not interchangeable:

```text
Lifetime
    Does the storage still exist?

Accessibility
    Is the CPU or GPU allowed to access it directly?

Synchronization
    Has the previous writer finished before the next reader begins?
```

For example, `std::shared_ptr` can keep a buffer alive. It does not make
simultaneous CPU and GPU access safe.

Likewise, a buffer can physically exist and remain owned by several handles
while still being inaccessible to the CPU because it uses private Metal
storage.

## What unified memory means on this Mac

A traditional discrete-GPU machine often looks conceptually like:

```text
CPU → system RAM

GPU → separate VRAM

data transfer crosses an interconnect
```

Apple Silicon looks conceptually like:

```text
CPU ─┐
     ├── one physical unified-memory system
GPU ─┘
```

The program prints Metal's:

```objective-c
device.hasUnifiedMemory
```

On the Apple M4 Pro it reports:

```text
yes
```

Unified physical memory does not mean:

```text
every std::vector is automatically a Metal buffer
every Metal buffer has a CPU pointer
CPU and GPU may write the same bytes simultaneously
synchronization is unnecessary
copies are never useful
```

Metal still needs a resource object describing the allocation:

```text
MTLBuffer
    size
    lifetime
    storage mode
    GPU binding
    access policy
```

## Storage mode is an access policy

This lesson uses two Metal storage modes:

| Storage mode | CPU direct access | GPU access | Typical flow |
|---|---:|---:|---|
| `Shared` | yes | yes | CPU write → GPU use → wait → CPU read |
| `Private` | no | yes | shared staging → copy → private → copy → shared staging |

On Apple Silicon, private storage should not be imagined as a physically
separate VRAM chip.

It means:

```text
Metal presents this resource as GPU-only.
The CPU cannot directly map and dereference its contents.
Metal may keep or arrange it in a GPU-oriented form.
```

Private storage is not guaranteed to be faster for every workload. A real
runtime measures the relevant models and hardware.

## Experiment A: shared storage

The buffers are allocated with:

```objective-c
id<MTLBuffer> shared_input =
    [device
        newBufferWithLength:byte_count
                   options:MTLResourceStorageModeShared];
```

A shared buffer exposes:

```objective-c
buffer.contents
```

which is a CPU pointer to its bytes.

The CPU initializes the input:

```cpp
float* values =
    static_cast<float*>(shared_input.contents);

values[0] = 1.0f;
values[1] = 2.0f;
```

The GPU receives the same Metal buffer:

```metal
device const float* input [[buffer(0)]]
```

and writes:

```metal
output[global_id] =
    input[global_id] * 3.0f + 1.0f;
```

No upload or download copy is required for this shared path.

## Shared does not mean simultaneous

The safe timeline is:

```text
CPU writes input and output sentinel values
        ↓
CPU encodes and commits the GPU command
        ↓
GPU reads input and writes output
        ↓
CPU waits for command-buffer completion
        ↓
CPU reads output
```

The important code is:

```objective-c
[command_buffer commit];
[command_buffer waitUntilCompleted];

// CPU reads output only here.
```

`commit()` starts asynchronous work. It does not mean the GPU has already
finished.

The lesson deliberately does not read the output between `commit()` and
`waitUntilCompleted()`. Such an observation would be an unsafe CPU/GPU access
race, not a meaningful test.

`waitUntilCompleted()` does not copy the shared buffer. It establishes an
execution handoff:

```text
before the wait completes:
    GPU may still be using the resource

after the wait completes:
    the submitted GPU work is finished
    CPU may safely inspect the result
```

## Experiment B: private storage

The two compute buffers use:

```objective-c
MTLResourceStorageModePrivate
```

Conceptually:

```text
private_input
private_output

CPU direct access: no
GPU access:        yes
```

The CPU therefore cannot do this:

```cpp
private_input.contents[index] = value; // wrong model
```

It also cannot directly read `private_output` after the GPU finishes.
Waiting changes completion state; it does not change the storage mode.

## Staging buffers

A staging buffer is a shared resource used to move data between the CPU and
GPU-only resources.

This lesson allocates:

```text
upload_staging     Shared
private_input      Private
private_output     Private
readback_staging   Shared
```

The upload flow is:

```text
ordinary CPU values
        ↓ CPU stores
shared upload_staging
        ↓ GPU copy
private_input
```

The readback flow is:

```text
private_output
        ↓ GPU copy
shared readback_staging
        ↓ after completion
ordinary CPU loads
```

This pattern appears throughout accelerator runtimes:

```text
host-visible staging
        ↕ copies
device-oriented working storage
```

## What is a blit?

In Metal, a blit command performs operations such as resource copies.

The buffer copy is encoded as:

```objective-c
id<MTLBlitCommandEncoder> encoder =
    [command_buffer blitCommandEncoder];

[encoder
    copyFromBuffer:source
       sourceOffset:0
           toBuffer:destination
  destinationOffset:0
               size:byte_count];

[encoder endEncoding];
```

This is GPU command encoding. It is not a CPU `memcpy`.

Creating a private buffer does not automatically populate it. The upload
copy is what places the input bytes into `private_input`.

Similarly, the download copy is what makes the private output available in
a CPU-accessible readback buffer.

## One ordered command buffer

The private experiment encodes:

```text
Command buffer
│
├── blit encoder:
│     upload_staging → private_input
│
├── compute encoder:
│     private_input → private_output
│
└── blit encoder:
      private_output → readback_staging
```

A Metal command buffer is an ordered list of GPU commands. The resources in
this lesson are allocated directly from the device and use Metal's default
hazard tracking. Metal therefore tracks the copy → compute → copy resource
dependencies used here.

Therefore the CPU does not wait between every stage:

```text
encode upload
encode compute
encode download
commit once
wait once
```

This would be unnecessarily restrictive:

```text
upload → CPU wait → compute → CPU wait → download → CPU wait
```

The GPU command list already expresses the dependency chain.

The final wait means that all earlier commands in that command buffer have
also completed.

More advanced runtimes can deliberately use untracked resources, especially
resources suballocated from heaps. Those runtimes must express the necessary
dependencies explicitly with mechanisms such as fences. Lesson 16 uses the
safe default tracked mode.

## The same kernel accepts both storage modes

The kernel declaration is:

```metal
device const float* input
device float* output
```

The shared experiment binds shared buffers.

The private experiment binds private buffers.

No second kernel is needed.

This reveals an overloaded word:

```text
MSL device pointer
    memory reachable through a Metal buffer from GPU code

MTLStorageModePrivate
    Metal resource not directly CPU-accessible
```

`device float*` does not mean the buffer must use private storage.

## Several meanings of shared and private

These terms appear in different layers:

| Term | Meaning |
|---|---|
| `std::shared_ptr` | several CPU handles jointly keep an object alive |
| `MTLStorageModeShared` | CPU and GPU may directly access the Metal resource |
| `threadgroup` memory | temporary scratch shared by lanes of one group |
| C++ local variable | normally private to one CPU function invocation |
| MSL local variable | normally private to one GPU thread |
| `MTLStorageModePrivate` | Metal resource is not directly CPU-accessible |

These meanings must not be mixed.

For example:

```text
private Metal buffer
    one complete GPU resource
    many GPU threads may access it

private lane variable
    one value belonging to one GPU thread
```

## Barrier versus completion wait

These synchronize different participants:

```text
threadgroup_barrier(...)
    called inside a kernel
    coordinates lanes in one threadgroup

waitUntilCompleted()
    called by the CPU
    waits for the complete submitted GPU command buffer
```

A threadgroup barrier does not tell the CPU that the kernel finished.

A CPU completion wait does not act as an intermediate barrier between lanes
inside a running kernel.

## Why this lesson does not call `synchronizeResource`

Metal also has a managed storage mode for macOS systems where the CPU and GPU
maintain separate copies of one resource.

That model uses operations such as:

```objective-c
didModifyRange(...)
synchronizeResource(...)
```

The current SDK recommends shared storage for the Apple Silicon path used by
this lesson. Managed-storage synchronization is not the general
synchronization mechanism for this program.

Important distinctions:

```text
synchronizeResource
    makes managed resource copies coherent

waitUntilCompleted
    waits until submitted execution finishes

private → shared blit
    copies GPU-only output into CPU-accessible storage
```

Waiting does not make a private buffer CPU-accessible.

Synchronizing a resource does not replace waiting for unfinished execution.

## How this relates to inference

A simplified inference memory flow can look like:

```text
model file or tokenizer input
        ↓ CPU prepares data
shared upload staging
        ↓
GPU-oriented model-weight buffers
GPU activation buffers
GPU KV-cache buffers
        ↓ many kernels reuse them
small result or token data
        ↓
shared readback
        ↓
CPU scheduler or sampler
```

Not every production runtime uses this exact storage choice for every
tensor. The key architectural questions remain:

```text
Where does this tensor live?
Who needs direct access?
Does it need an upload or readback?
Which command writes it?
Which later command reads it?
What establishes the dependency?
How long must the storage remain alive?
```

These questions later apply to:

```text
model weights
temporary activations
attention scores
KV-cache pages
batch metadata
sampled token output
```

## Why Lesson 14 is not changed

Lesson 14 intentionally presents a very small MiniGPU API:

```cpp
Buffer buffer = device.allocate(count);
buffer.store(index, value);
float value = buffer.load(index);
```

Its implementation always allocates shared Metal storage. Supporting private
buffers correctly would require adding:

```text
storage-mode metadata
CPU-access validation
copy commands
copy-event lifetime tracking
more general queue commands
```

Those are reasonable future runtime features, but adding all of them here
would hide the memory concepts behind abstraction machinery.

Lesson 16 therefore temporarily looks beneath MiniGPU and uses native Metal,
as Lesson 13 did.

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
  outputs/infer-lab/phase0/lesson16-real-gpu-memory/demo.mm \
  -framework Foundation \
  -framework Metal \
  -o work/lesson16_real_gpu_memory

./work/lesson16_real_gpu_memory
```

## Expected shape of the output

Exact formatting may vary:

```text
Metal device: Apple M4 Pro
Unified physical memory: yes

Experiment A: shared storage
CPU direct access: yes
Before submission: [-1, -1, -1, -1, -1, -1, -1, -1]
After completion: [4, 7, 10, 13, 16, 19, 22, 25]
Path: CPU write -> GPU compute -> wait -> CPU read

Experiment B: private storage with staging
Private input CPU direct access: no
Private output CPU direct access: no
Readback before submission: [-1, -1, -1, -1, -1, -1, -1, -1]
Readback after completion: [4, 7, 10, 13, 16, 19, 22, 25]
Path: shared upload -> blit -> private compute -> blit -> shared readback

PASS
```

## Main takeaways

1. Unified memory describes the physical memory architecture.
2. A Metal storage mode describes a resource's access policy.
3. Shared buffers support direct CPU and GPU access.
4. Shared buffers still require synchronization between CPU and GPU use.
5. Private buffers are GPU-accessible but not directly CPU-accessible.
6. Shared staging buffers upload data to and read data from private buffers.
7. Blit encoders record GPU resource-copy commands.
8. One ordered command buffer can express copy → compute → copy.
9. A final completion wait makes subsequent CPU readback safe.
10. Waiting does not make private memory CPU-accessible.
11. MSL `device` address space and Metal private storage are different ideas.
12. Lifetime, accessibility and synchronization solve different problems.

## What to understand versus skim

Understand:

```text
Shared: CPU yes, GPU yes
Private: CPU no, GPU yes
staging: shared bridge around private storage
copy in → compute → copy out
commit starts asynchronous execution
wait creates the CPU/GPU handoff
```

Skim:

```text
Objective-C error conversion
null-object checks
printing helpers
vector comparison
pipeline compilation boilerplate
```

## Where the roadmap goes next

Lessons 15 through 19 now cover:

```text
launch geometry
shared and private GPU resources
CPU/GPU synchronization
threadgroup memory and barriers
atomics and races
multi-pass reductions
stable softmax
basic performance measurement
```

The next layer can move toward:

```text
compiler IR and Metal source generation
MiniTensor shapes, strides and dtypes
reusable tensor operations
```
