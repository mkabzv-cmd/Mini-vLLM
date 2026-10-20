# Lesson 12: When threads collide — races and atomics

Lesson 11 taught threads how to **meet** at a barrier. Lesson 12 teaches
threads how to **change the same value without overwriting one another**.

We extend MiniGPU with:

- deterministic shared-memory race detection,
- a threadgroup `shared_atomic_add`,
- a device-visible `AtomicU32Buffer`,
- atomic `load`, `store`, `fetch_add`, and `compare_exchange`,
- atomic-operation counting for contention experiments,
- a timeout wrapper that turns accidental deadlocks into test failures.

Everything still executes on the Mac CPU. The purpose is to make accelerator
semantics inspectable before we map them onto Apple Metal.

## Learning goals

By the end of this lesson, you should be able to:

1. recognize a data race,
2. distinguish a data race from a broader race condition,
3. explain a lost update step by step,
4. explain why a barrier does not make an increment atomic,
5. explain why atomic load followed by atomic store is still not atomic RMW,
6. use `fetch_add` to create a counter or unique tickets,
7. explain the purpose of compare-and-exchange,
8. distinguish threadgroup, device, and system atomic scopes,
9. understand relaxed versus acquire/release ordering at a beginner level,
10. explain why correct atomics can still be slow,
11. reduce contention through hierarchical aggregation,
12. identify inference-engine problems where atomics are useful.

## Where this fits

```text
MiniGPU
├── buffers and ownership                  Lesson 10
├── asynchronous queue and events          Lesson 10
├── threadgroup memory and barriers        Lesson 11
├── parallel reduction                     Lesson 11
├── race detection                         this lesson
├── atomic read-modify-write               this lesson
├── first Metal backend                    next major step
├── kernel compiler                        later
└── optimized kernel library               later

MiniTensor
└── tensor operations

TinyLLM
└── transformer inference

Mini-vLLM
└── batching, KV paging, scheduling

Mini-Dynamo
└── multi-worker and multi-node serving
```

Race reasoning will reappear at every layer: inside kernels, inside KV-cache
allocators, inside schedulers, and across distributed workers.

## 1. What is a data race?

Two logical threads have a potential data race when:

1. they access the same memory location,
2. at least one access writes,
3. and no valid synchronization orders those accesses.

Examples:

| Pattern | Race? | Reason |
|---|---:|---|
| Two lanes read `weights[5]` | No | Neither changes the value |
| Lane `i` writes `output[i]` | No | Lanes use different locations |
| Two lanes write `output[0]` | Yes | Competing writes |
| One lane reads while another writes | Yes | Reader may observe old or new state |
| All lanes use atomic add | No data race | Each RMW is indivisible |
| Producer writes, barrier, consumer reads | No | The barrier separates phases |

Even two threads writing the same value is still a race. “The answer happens
to be unchanged” is not a synchronization rule.

## 2. Data race versus race condition

The phrases are related, but not identical:

- A **data race** is an unsafe conflicting memory-access pattern.
- A **race condition** is any bug where correctness depends on unpredictable
  ordering.

This lesson contains an important example that has no individual data race
but still has a race condition:

```cpp
std::uint32_t old = counter.load(0);
counter.store(0, old + 1);
```

Our `AtomicU32Buffer::load` and `store` are individually atomic. However, the
pair is not one indivisible operation. Another lane may act between them.

This distinction matters in production: replacing ordinary variables with
atomic variables does not automatically make a multi-step algorithm correct.

## 3. A deterministic lost update

Start with:

```text
counter = 0
```

Eight lanes execute:

```cpp
std::uint32_t old = counter.load(0);
context.barrier(600);

counter.store(0, old + 1);
context.barrier(601);
```

The first barrier deliberately forces this schedule:

```text
lane 0 loads 0
lane 1 loads 0
lane 2 loads 0
...
lane 7 loads 0
barrier

lane 0 stores 1
lane 1 stores 1
lane 2 stores 1
...
lane 7 stores 1
barrier
```

Mathematical expectation:

```text
0 + eight increments = 8
```

Actual final value:

```text
1
```

Seven updates disappeared. Each lane calculated a replacement from the same
old snapshot.

The barrier did exactly what we asked: everyone completed the loads before
anyone performed the stores. It did not merge the eight updates. In fact, it
made the lost update deterministic.

An analogy is eight clerks:

1. every clerk reads an account balance of zero,
2. every clerk independently adds one on paper,
3. every clerk replaces the account balance with one.

No clerk saw another clerk's update.

## 4. Atomic read-modify-write

We need the three logical steps to become one indivisible operation:

```text
read old value
calculate old + delta
write new value
```

MiniGPU exposes:

```cpp
std::uint32_t old = counter.fetch_add(0, 1);
```

Conceptually, no other atomic operation on this location can appear inside
that read-modify-write operation.

One possible execution is:

```text
lane 6 receives old value 0, counter becomes 1
lane 2 receives old value 1, counter becomes 2
lane 7 receives old value 2, counter becomes 3
...
counter becomes 8
```

We cannot predict which lane receives which old value. We can prove:

- the final counter is eight,
- every returned old value is unique,
- sorting the returned values produces `0, 1, 2, ..., 7`.

The imaginary instant at which an atomic operation takes effect is called
its **linearization point**. Operations may physically overlap, but their
effects can be understood as one valid serial order.

## 5. Atomic load plus atomic store is not atomic increment

This is incorrect:

```cpp
auto old = counter.load(0);     // atomic operation A
counter.store(0, old + 1);      // atomic operation B
```

There is a gap between A and B.

This is correct:

```cpp
counter.fetch_add(0, 1);        // one atomic RMW operation
```

Memorable rule:

> Individually atomic steps do not make the entire sequence atomic.

The same issue appears with code such as:

```text
if queue is not empty:
    remove an item
```

An atomic check and an atomic removal do not make the combined check-then-act
sequence indivisible.

## 6. What `fetch_add` returns

`fetch_add` returns the value from before the addition:

```cpp
slot = next_slot.fetch_add(0, 1);
```

If many lanes execute this, they receive unique slot numbers. This is useful
for compacting irregular results:

```cpp
if (item_is_valid) {
    slot = valid_count.fetch_add(0, 1);
    compacted_output[slot] = item;
}
```

It is also useful for mixture-of-experts routing:

```cpp
slot = expert_count[expert].fetch_add(1);

if (slot < expert_capacity) {
    expert_tokens[expert][slot] = token;
}
```

The increment counts the token and the returned old value reserves a unique
position.

## 7. Compare-and-exchange

MiniGPU also implements:

```cpp
bool won = value.compare_exchange(
    index,
    expected,
    desired
);
```

Its meaning is:

```text
if current value equals expected:
    replace it with desired
    return true
else:
    copy current value into expected
    return false
```

If sixteen lanes all try to replace zero with their own nonzero ID, exactly
one wins. We cannot predict the winning lane.

Compare-and-exchange—often abbreviated **CAS**—is a foundation for:

- one-time initialization,
- lock-free queues,
- reference counts,
- free lists,
- memory allocators,
- KV-cache page management,
- state machines.

CAS loops, the ABA problem, and genuinely lock-free structures are deferred
until our runtime needs them.

## 8. Barriers and atomics solve different problems

| Barrier | Atomic operation |
|---|---|
| Coordinates a group of lanes | Protects one memory operation/location |
| Creates execution phases | Creates an indivisible update |
| Every participating lane must arrive | Only accessing lanes participate |
| Makes earlier writes visible later | Does not make the group meet |
| Works within one threadgroup | May have threadgroup/device/system scope |
| Useful for reductions and tiled algorithms | Useful for counters and reservations |

Memorable rule:

> A barrier answers “has everyone reached this phase?”  
> An atomic answers “who changes this value first?”

Atomics are not barriers. Consider:

```cpp
context.shared_atomic_add(0, 1.0f);

if (context.local_id() == 0) {
    use(context.shared_load(0));
}
```

Every atomic addition itself is safe, but lane zero may read before every
other lane has added. Our beginner race model rejects the unsynchronized
atomic/ordinary access.

The correct pattern is:

```cpp
context.shared_atomic_add(0, 1.0f);
context.barrier(7);

if (context.local_id() == 0) {
    use(context.shared_load(0));
}
```

## 9. The race detector we built

Using real CPU scheduling to detect a simulated GPU race would be unreliable:

```text
run 1: CPU threads overlap → bug appears
run 2: CPU threads serialize → bug disappears
```

Instead, MiniGPU reasons about **barrier phases**:

```text
phase 0: kernel start → first barrier
phase 1: first barrier → second barrier
phase 2: second barrier → third barrier
...
final phase: last barrier → group completion
```

For every threadgroup-memory index, the simulator records which lanes
performed:

- ordinary loads,
- ordinary stores,
- atomic additions.

At a barrier, every lane is stopped. The simulator examines the completed
phase before releasing the group.

The rules are:

```text
read  + read      safe
atomic + atomic   safe

read  + store     race
store + store     race
read  + atomic    race in our beginner model
store + atomic    race
```

Only cross-lane conflicts count. A single lane's own operations are ordered
by its program.

If a kernel has no final barrier, the detector checks the final phase when
the last lane exits.

## 10. Deterministic diagnostics

Many lanes may participate in the same race. Reporting “whichever CPU thread
arrived second” would produce unstable tests.

Our detector chooses one canonical conflict in this order:

1. lowest memory index,
2. lowest first lane ID,
3. lowest second lane ID,
4. fixed access-kind order.

A `SharedMemoryRaceError` exposes structured fields:

```cpp
error.group_id();
error.phase();
error.index();
error.first_local_id();
error.first_access();
error.second_local_id();
error.second_access();
```

This is similar to building a tiny correctness tool inside a compiler/runtime
stack. Production GPU race checkers are much more sophisticated, but the core
idea is already visible:

```text
record memory accesses
derive happens-before relationships
report unordered conflicts
```

## 11. Logical GPU race versus real C++ race

The CPU simulator protects its internal arrays with mutexes. Therefore, the
deliberately broken kernels do not create undefined C++ behavior.

We have two different checkers:

```text
ThreadSanitizer
└── detects races in our CPU simulator implementation

MiniGPU SharedMemoryRaceError
└── detects races in the GPU program being simulated
```

Both should pass simultaneously:

- ThreadSanitizer should report no host implementation race.
- MiniGPU should deliberately report the invalid simulated kernel.

This separation is essential when building emulators, virtual machines,
compilers, and hardware simulators.

## 12. Atomic scope

Atomicity must include every participant that can touch the location.

Vendor-neutral scopes are:

### Threadgroup scope

Participants:

```text
threads inside one group
```

Example:

```cpp
context.shared_atomic_add(0, 1.0f);
```

The location is private to one group's shared scratch memory.

### Device scope

Participants:

```text
all threadgroups on one accelerator
```

Our `AtomicU32Buffer` models a device-visible atomic location:

```cpp
global_counter.fetch_add(0, 1);
```

Every group sharing the handle observes one atomic order.

### System scope

Participants may include:

```text
CPU + accelerator + possibly other devices
```

System-scope synchronization requires platform support, coherent memory, and
careful memory ordering. Initially, our CPU and GPU will coordinate through
queues and events rather than both modifying the same atomic location.

Use the narrowest scope that contains every required participant. Wider
scope can cost more.

## 13. Memory order, simplified

Atomicity answers:

```text
is this operation indivisible?
```

Memory order answers:

```text
how are surrounding ordinary memory operations ordered?
```

Beginner meanings:

| Order | Meaning |
|---|---|
| Relaxed | The atomic value is correct; it does not publish surrounding data |
| Release | Publish ordinary writes that happened before the atomic |
| Acquire | After observing the atomic, see data published before it |
| Acquire-release | Perform both roles in one RMW operation |
| Sequentially consistent | Provide one strong shared ordering model |

Publication example:

```cpp
// Producer
payload = 42;
ready.store(1, release);

// Consumer
if (ready.load(acquire) == 1) {
    use(payload);
}
```

The release/acquire pair publishes the payload through the ready flag.

This lesson's CPU backend intentionally implements one strong, simple total
order for `AtomicU32Buffer`. We do not yet expose memory-order arguments.
That lets us learn atomicity without pretending we have implemented a full
portable GPU memory model.

Relaxed, acquire, and release operations will be added when we build a real
producer/consumer protocol.

## 14. Why atomics can be slow

Atomic means correct, not free.

Imagine:

```text
64 people
one ticket machine
```

Every person receives a unique ticket, but the hot machine becomes a
serialization point.

Atomics work well when:

- collisions are rare,
- updates are spread across many addresses,
- there are few updates,
- a unique reservation is required.

They work poorly when thousands of lanes repeatedly update one hot address.

## 15. Hierarchical aggregation

Our tests compare two ways for 32 lanes to count:

### Naive

```text
every lane → one device-wide atomic add

32 lanes
32 global atomic RMW operations
```

### Hierarchical

```text
each group counts in threadgroup memory
one leader per group updates the global counter

4 groups × 8 lanes
4 global atomic RMW operations
```

Both produce 32. The hierarchical version reduces the number of global
atomic RMW operations by 8×. That is the structural reason it can reduce
contention; this CPU simulator is not a GPU performance measurement.

The general pattern is:

```text
lane-private work
        ↓
threadgroup shared aggregation
        ↓
one partial result per group
        ↓
one device merge or another reduction pass
```

Lesson 11's reduction often beats a hot atomic because it distributes work
across a tree instead of serializing every lane at one location.

## 16. Floating-point atomic addition

Our `shared_atomic_add` uses FP32 because the existing teaching scratch
memory stores floats.

Even when every addition is preserved:

```text
(a + b) + c
```

may not have exactly the same FP32 result as:

```text
a + (b + c)
```

Atomic execution order can vary, so the low bits of a floating-point sum can
vary without any data race.

Therefore:

- race freedom does not imply bitwise reproducibility,
- tests use small integers exactly representable in FP32,
- important reductions often use structured trees and higher-precision
  accumulation.

## 17. Why cross-group spin barriers are dangerous

Do not implement this inside a kernel:

```text
every group increments arrived
every group spins until arrived == group_count
```

An accelerator may have room to run only some groups simultaneously:

```text
resident groups occupy all execution resources
        ↓
they spin waiting for nonresident groups
        ↓
nonresident groups cannot start
        ↓
deadlock
```

Use:

- a kernel boundary,
- a queue dependency,
- an event,
- or a platform-supported cooperative launch.

Our multi-pass reduction uses kernel boundaries for exactly this reason.

## 18. Relevance to inference engineering

### Mixture-of-experts routing

Tokens need unique slots in expert buffers:

```cpp
slot = expert_count[expert].fetch_add(1);
```

A popular expert becomes a contention hotspot, motivating grouping,
prefix-sum routing, or hierarchical counting.

### KV-cache allocation

Paged KV caches need:

- page reference counts,
- free-list updates,
- allocation state transitions,
- request ownership,
- safe reclamation.

Atomic counters and CAS are common building blocks.

### Compaction

Variable numbers of valid items can reserve output positions with
`fetch_add`.

### Sparse scatter-add

Several lanes may contribute to one output index. Atomic addition prevents
lost contributions, although floating-point order can affect reproducibility.

### Histograms and sampling

Token counts and bucket histograms may use local atomic bins followed by a
merge.

### Softmax and normalization

Do not have every lane atomically add into one hot global sum. Threadgroup
reduction is generally the better structure.

### CPU serving runtime

The same concepts appear outside GPU kernels:

- scheduler queue indices,
- request state,
- reference counts,
- metrics,
- work stealing,
- lock-free data structures.

The hardware API changes; concurrency reasoning transfers.

## 19. Runtime implementation map

### `AtomicU32Buffer`

The CPU backend stores U32 elements behind shared storage and a mutex.

```text
load              one indivisible operation
store             one indivisible operation
fetch_add         one indivisible read-modify-write
compare_exchange  one indivisible conditional RMW
```

Copied handles share the same storage and atomic order.

`rmw_operation_count()` lets tests compare algorithms structurally. It is
debug instrumentation, not a production performance counter.

The CPU implementation uses one mutex for the entire `AtomicU32Buffer`, so
operations on different indices also serialize here. This gives correct
functional semantics but does not model real per-address atomic throughput.

### `shared_atomic_add`

The existing per-group scratch array gains one indivisible fetch-add
operation. Multiple atomic additions to the same index in one phase are
legal.

The global API uses the conventional name `fetch_add`; the context API uses
the CUDA-like spelling `shared_atomic_add` to emphasize the memory space.
Both return the old value.

### Phase race detector

Access bitmasks fit in 64 bits because the teaching simulator caps each
group at 64 lanes.

### Barrier integration

The last lane arriving at a barrier closes and validates the phase before
any lane is released.

### Final-phase validation

The last exiting lane validates accesses made after the final barrier.

### Error propagation

A detected race:

```text
breaks the threadgroup
→ wakes barrier waiters
→ finishes the command with an exception
→ Event::wait() throws SharedMemoryRaceError
→ the queue remains usable
```

If independent failures occur simultaneously, the first failure observed by
the group becomes the command's error. We provide deterministic selection
inside one completed shared-memory race phase, but we do not promise a fixed
priority between unrelated exceptions, early barrier divergence, and races.

### `ThreadContext` lifetime

`ThreadContext` is valid only during its kernel callback. It cannot be
copied or moved. Never save its address, return a reference to it, or use it
from another host thread—even while the callback is active—or after the
callback returns. Its internal runtime state is stack-local to one simulated
lane.

## 20. Build and run

From this directory:

```bash
clang++ \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -pthread \
  minigpu.cpp \
  tests.cpp \
  -o minigpu_tests

python3 run_tests.py ./minigpu_tests
```

The Python wrapper enforces a 30-second process timeout. If a negative
barrier test ever deadlocks, the test command terminates instead of hanging
forever.

The timeout can be increased for a slow sanitizer machine:

```bash
MINIGPU_TEST_TIMEOUT=60 \
  python3 run_tests.py ./minigpu_tests_tsan
```

Expected final line:

```text
All MiniGPU atomics and race tests passed
```

### AddressSanitizer and UndefinedBehaviorSanitizer

```bash
clang++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  -pthread \
  minigpu.cpp \
  tests.cpp \
  -o minigpu_tests_sanitized

python3 run_tests.py ./minigpu_tests_sanitized
```

### ThreadSanitizer

```bash
clang++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fsanitize=thread \
  -fno-omit-frame-pointer \
  -pthread \
  minigpu.cpp \
  tests.cpp \
  -o minigpu_tests_tsan

TSAN_OPTIONS=halt_on_error=1 \
  python3 run_tests.py ./minigpu_tests_tsan
```

## 21. What the tests prove

The complete regression suite retains Lesson 11's barriers and reductions,
then proves:

1. atomic load plus atomic store can deterministically lose updates,
2. `fetch_add` produces the correct final counter,
3. returned tickets are unique but not assigned in lane order,
4. compare-exchange produces exactly one winner,
5. failed compare-exchange updates `expected` without changing the value,
6. non-atomic shared RMW is reported at a deterministic phase,
7. direct read/write conflicts are detected,
8. race diagnostics are canonical and repeatable,
9. a race error propagates through `Event::wait()`,
10. the queue recovers after a race,
11. shared atomic additions preserve all updates,
12. lane 63 is handled safely by the 64-bit access masks,
13. shared atomic old values form unique tickets,
14. atomic addition does not act like a barrier,
15. mixed atomic/ordinary access is rejected,
16. read/read sharing is legal,
17. distinct-index writes are legal,
18. repeated same-lane access is legal,
19. barrier-separated accesses are legal,
20. a race after the final barrier is still detected,
21. an atomic-only final phase is legal,
22. shared-atomic bounds errors preserve their original exception,
23. hierarchical aggregation preserves the answer,
24. hierarchical aggregation performs fewer global atomic RMWs,
25. the existing reduction produces no race false positive.

## 22. Suggested reading order

1. Read the lost-update and ticket tests.
2. Read the `AtomicU32Buffer` public API.
3. Read its mutex-backed CPU implementation.
4. Read `shared_atomic_add`.
5. Read `SharedMemory::close_phase`.
6. Read the canonical conflict scanner.
7. Read how the last barrier arrival validates a phase.
8. Read final-phase validation in `finish`.
9. Read the queue only after those concepts are clear.

## 23. Exercises

### Exercise A: predict the lost update

Two lanes execute:

```text
load counter
add one locally
store counter
```

Write one schedule that ends at two and one schedule that ends at one.

### Exercise B: classify access patterns

Label each safe, racy, or atomic-but-contended:

```text
every lane reads weights[5]
every lane writes output[global_id]
every lane writes output[0]
every lane calls fetch_add(counter, 1)
lane 0 writes scratch, barrier, lane 1 reads scratch
```

### Exercise C: repair the counter

Replace load/barrier/store with `fetch_add`. Save the returned old values and
verify the sorted tickets form a consecutive range.

### Exercise D: choose a scope

Choose threadgroup, device, or system scope:

```text
one group's scratch counter
one counter shared by every GPU group
a CPU and GPU concurrently updating shared state
```

### Exercise E: calculate contention reduction

For 4,096 lanes in groups of 64:

```text
naive global atomics              = ?
one global atomic per group       = ?
reduction factor                  = ?
```

Answers:

```text
4096, 64, 64×
```

### Exercise F: implement shared atomic maximum

Add a shared atomic maximum for integer data. Test that:

- every candidate participates,
- the final maximum is correct,
- the winning lane is not assumed,
- ordinary access is separated by a barrier.

### Exercise G: inspect a race report

Change the flawed shared RMW test so the collision occurs at index two.
Verify the structured error fields change accordingly.

### Exercise H: MoE routing

Explain:

1. what breaks if expert slot allocation uses load/add/store,
2. what atomics repair,
3. what performance problem remains if every token selects one expert.

## Checkpoint questions

You understand this lesson when you can answer:

1. What three conditions create a data race?
2. What is the difference between a data race and a race condition?
3. Why do individually atomic load and store calls not make an increment
   atomic?
4. What exactly is a lost update?
5. Why can a barrier make the lost update deterministic?
6. What does `fetch_add` return?
7. Why are its tickets unique but unpredictably assigned?
8. What does compare-and-exchange do when the expected value is wrong?
9. What problem does a barrier solve that an atomic does not?
10. What problem does an atomic solve that a barrier does not?
11. What is an atomic scope?
12. When is relaxed ordering sufficient?
13. What do release and acquire publish?
14. Why can a correct atomic operation still become a bottleneck?
15. How does hierarchical aggregation reduce contention?
16. Why does our race detector reason about phases instead of wall-clock
    overlap?
17. Why should ThreadSanitizer remain clean while a simulated kernel raises
    `SharedMemoryRaceError`?
18. Why can FP atomic addition be race-free but not bitwise deterministic?
19. Why is a cross-group spin barrier dangerous?
20. Which inference-runtime components need counters, CAS, or reference
    counts?

## Deliberately deferred

- A complete formal memory model.
- Configurable relaxed/acquire/release operations.
- Global-buffer race detection.
- Atomic minimum/maximum and bitwise operations.
- CAS retry loops and the ABA problem.
- Lock-free queues and allocators.
- SIMD-group atomics.
- Backend-specific Metal atomic capabilities.
- Real accelerator contention measurements.

The next major step is to map the stable concepts—buffers, dispatches,
thread IDs, barriers, shared memory, reduction, and atomics—onto our first
real Metal compute backend on the Apple GPU.
