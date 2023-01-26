# FIL Implementation
This document is intended to provide additional detail about this
implementation of FIL to help guide future FIL contributors. Because this is
the first cuML algorithm to attempt to provide a unified CPU/GPU codebase that
does *not* require nvcc, CUDA or any other GPU-related library for its CPU-only
build, we also go over general strategies for CPU/GPU interoperability as used
by FIL.

**A NOTE ON THE `kayak` NAMESPACE:** In addition to FIL-specific code, the new
implementation requires some more general-purpose CPU-GPU interoperable
utilities. Many of these utilities are either already implemented in RAFT (but
do not provide the required CPU-interoperable compilation guarantees) or are a
natural fit for incorporation in RAFT. In order to allow for more careful
integration with the existing RAFT codebase and interoperability
strategies, these utilities are currently provided in the `kayak`
namespace but will be moved into RAFT over time. Other algorithms should
not make use of the `kayak` namespace but instead wait until this
transition has taken place.

## Design Goals
1. Provide state-of-the-art runtime performance for forest models on GPU,
   especially for cases where CPU performance will not suffice (e.g. large
   batches, deep trees, many trees, etc.).
2. Ensure that the public API is the same for both CPU and GPU execution.
3. Re-use as much code as possible between CPU and GPU implementations.
4. Provide near-state-of-the-art runtime performance for forest models on most
   CPUs without vendor-specific optimizations.

## Strategies for CPU/GPU code re-use

This FIL implementation now makes use of a build-time variable
`ENABLE_GPU` to determine whether or not to compile CUDA code. If
`ENABLE_GPU` is not set, FIL is guaranteed to compile without nvcc, access
to CUDA headers, or any other GPU-related library.

We explicitly wish to avoid excessive use of `#ifdef` statements based on this
variable, however. Interleaving CPU and GPU code via `#ifdef` branches both
reduces readability and discourages writing of truly interoperable code.
Ideally, `#ifdef` statements should be used solely and sparingly for
conditional header inclusion. This presents additional challenges but also
opportunities for a cleaner implementation of a unified CPU/GPU
codebase.

It is also occasionally useful to make use of a `constexpr` value
indicating whether or not `ENABLE_GPU` is set, which we introduce as
`kayak::GPU_ENABLED`.

### Avoiding CUDA symbols in CPU-only builds
The most significant challenge of attempting to create a unified CPU/GPU
implementation is ensuring that no CUDA symbols are exposed in the CPU-only
build. To illustrate the general strategy, we will look at a specific example:
the implementation of the main inference loop. Code for this loop is provided
in the following four files:

```
detail/
├─ infer.hpp  # "Consumable" header
├─ infer/     # "Implementation" directory
│  ├─ cpu.hpp
│  ├─ gpu.cuh
│  ├─ gpu.hpp
```

For brevity, we introduce the concepts of "consumable" and "implementation"
headers. Consumable headers can be included in any other header and are
guaranteed not to themselves include any header with CUDA symbols
`ENABLE_GPU` is not defined.
Implementation headers can *only* be included by their associated consumable
header or directly in a source file. They should *never* be directly included
by any other consumable header except their own.

By creating a clear separation of these two header types, we guarantee that any
source file that includes a consumable header should be compilable with or
without access to CUDA headers. Note that all public headers should be
consumable, but not all consumable headers need be made public. In the
particular example under consideration, `infer.hpp` is consumable, but we keep
it in the detail directory to indicate that it is not part of the public API.

Let's take a closer look at each of the "infer" headers. `infer.hpp`
implements `detail::infer`, a function templated on both the execution device
type (`D`) and the type of the forest model being evaluated `forest_t`.
If we were to look at the implementation of this template, we would note
that there is no code specialized for either possible value of `D`. At the
level of consumable headers, we have abstracted away the difference between
GPU and CPU in order to ensure that this template is completely reusable
between GPU and CPU.

Where we _need_ to provide distinct logic between GPU and CPU
implementations, we do so in implementation headers. In `infer/cpu.hpp`, we
have a fully-defined template for CPU specializations of
`detail::inference::infer`. If `kayak::GPU_ENABLED` is `true`, we also include the
failure case for the GPU specialization. In `infer/gpu.hpp` we *declare* but do
not *define* the GPU specializations. In `infer/gpu.cuh` we provide the
definition for the GPU specializations.

`infer.hpp` includes `infer/cpu.hpp` and `infer/gpu.hpp`, but *not*
`infer/gpu.cuh`. Instead, `infer/gpu.cuh` is included directly in the CUDA
source files that require access to the full definition.

Structuring the code in this way, we have a single separation point between
code that will and will not compile without access to CUDA headers. A similar
approach is used anywhere else in the implementation where we need distinct
logic for CPU and GPU. Otherwise, we are free to use anything defined in a
consumable header without worrying about whether the current translation unit
will ultimately be compiled with gcc or nvcc or whether our current build does
or does not have GPU enabled.

### Re-using code

Ultimately, many GPU and parallel CPU algorithms do not differ much in their
actual steps, but optimizing each requires careful attention to the
differing parallelism models and memory access models on each hardware
type. This means that with a little care, we can separate details related
to parallelism and memory access from the actual algorithm logic. This logic
will be the same for both CPU and GPU, but the now-isolated parallelism
and memory access code can be independently optimized.

The process of actually performing this separation usually starts by
identifying the basic single "task" that each parallel worker must take on.
It is not always entirely obvious how granular this task should be. For
instance, in the case of forest models, we might consider the basic task to
be evaluating a single row with all trees in the forest, evaluating all rows
with a single tree of the forest, evaluating a single row with a single
tree, evaluating a single node of a tree on a single row, evaluating a sub-tree
of a specific size on a single row, etc.

In order to offer optimal performance on the widest range of models, the
present implementation defines the underlying worker task as evaluating
a single row on a single tree, but specific model characteristics (e.g. very
small or large trees) might benefit from other task granularity.

Once we have identified the underlying task, we implement this
directly in a way that is independent of the parallelism model or memory
access patterns. That is to say, we assume that we are already executing
on a single worker and that the memory is arranged optimally for this task. In
the current implementation, this task is defined in
`detail/evaluate_tree.hpp`.

Looking at this header, we should note that there is no logic specific to the
GPU or CPU. Instead we defer this to `infer_kernel`, which provides
