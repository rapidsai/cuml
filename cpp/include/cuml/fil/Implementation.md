# FIL Implementation
This document is intended to provide additional detail about this
implementation of FIL to help guide future FIL contributors. Because this is
the first cuML algorithm to attempt to provide a unified CPU/GPU codebase that
does *not* require nvcc, CUDA or any other GPU-related library for its CPU-only
build, we also go over general strategies for CPU/GPU interoperability as used
by FIL.

**A NOTE ON THE `raft_proto` NAMESPACE:** In addition to FIL-specific code, the new
implementation requires some more general-purpose CPU-GPU interoperable
utilities. Many of these utilities are either already implemented in RAFT (but
do not provide the required CPU-interoperable compilation guarantees) or are a
natural fit for incorporation in RAFT. In order to allow for more careful
integration with the existing RAFT codebase and interoperability
strategies, these utilities are currently provided in the `raft_proto`
namespace but will be moved into RAFT over time. Other algorithms should
not make use of the `raft_proto` namespace but instead wait until this
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
`CUML_ENABLE_GPU` to determine whether or not to compile CUDA code. If
`CUML_ENABLE_GPU` is not set, FIL is guaranteed to compile without nvcc, access
to CUDA headers, or any other GPU-related library.

We explicitly wish to avoid excessive use of `#ifdef` statements based on this
variable, however. Interleaving CPU and GPU code via `#ifdef` branches both
reduces readability and discourages writing of truly interoperable code.
Ideally, `#ifdef` statements should be used solely and sparingly for
conditional header inclusion. This presents additional challenges but also
opportunities for a cleaner implementation of a unified CPU/GPU
codebase.

It is also occasionally useful to make use of a `constexpr` value
indicating whether or not `CUML_ENABLE_GPU` is set, which we introduce as
`raft_proto::GPU_ENABLED`.

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
if `CUML_ENABLE_GPU` is not defined.
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
`detail::inference::infer`. If `raft_proto::GPU_ENABLED` is `false`, we also
include the GPU specializations, which will simply throw an exception if
invoked. In `infer/gpu.hpp` we *declare* but do not *define* the GPU
specializations. In `infer/gpu.cuh` we provide the full working definition for
the GPU specializations.

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
GPU or CPU. Instead we defer this to `infer_kernel`, which specifies how our
fundamental task gets assigned to individual "workers" (CPU threads for the CPU
or CUDA threads for the GPU). This is not a necessary constraint (i.e. we could
refactor later for CPU and GPU specific versions of `evaluate_tree`), but
re-using code in this way and providing a clean separation from the parallelism
model does offer advantages.

Beyond just the reduced maintenance of a single codebase and more modular
design, this gives us the opportunity to benefit from improvements in the CPU
implementation on GPU and vice versa. For instance, during the initial
development, only CPU tests were used to check for correctness, but GPU results
were shown to be correct as soon as they were added to the tests. Similarly,
during optimization, only GPU runtime and instructions were analyzed, but the
process of optimizing for the GPU resulted in significant speedups (over 50% on
a standard benchmark) on the CPU.

## Code Walkthrough

With some motivation for the general approach to CPU-GPU interoperability, we
now offer an overview of the layout of the codebase to help guide future
improvements. Because `raft_proto` utilities are going to be moved to RAFT or other
general-purpose libraries, we will not review anything within the `raft_proto`
directory here.

### Public Headers
* `constants.hpp`: Contains constant values that may be useful in working
  with FIL in other C++ applications
* `decision_forest.hpp`: Provides `decision_forest`, a template which provides
  concrete implementations of a decision forest. Because different types may
  be optimal for different sizes of models or models with different features,
  we implement this template on many different combinations of template
  parameters. This is provided in a public header in case other
  applications have more specialized use cases and can afford to work directly
  with this concrete underlying object.
* `forest_model.hpp`: Provides `forest_model`, a wrapper for a `std::variant`
  of all `decision_forest` implementations. This wrapper handles
  dispatching `predict` calls to the right underlying type.
* `exceptions.hpp`: Provides definitions for all custom exceptions that
  might be thrown within FIL and need to be handled by an external
  application.
* `postproc_ops.hpp`: Provides enums used to specify how leaf outputs should be
  processed.
* `treelite_importer.hpp`: Provides `import_from_treelite_model` and
  `import_from_treelite_handle`, either of which can be used to convert a
  Treelite model to a `forest_model` object to be used for accelerated
  inference.

### Detail Headers
* `cpu_introspection.hpp`: Provides constants and utilities to evaluate
  CPU capabilities for optimized performance.
* `decision_forest_builder.hpp`: Provides generic tools for building
  FIL forests from some other source. In the current FIL codebase, the
  Treelite import code is the only place this is used, but it could be used
  to create import utilities for other sources as well.
* `device_initialization.hpp`: Contains code for anything that must be done
  to initialize execution on a device. For GPUs, this may mean setting
  specific CUDA options.
* `evaluate_tree.hpp`: Contains code for evaluating a single tree on input
  data.
* `forest.hpp`: Provide the storage struct `forest` whose *sole*
  responsibility is to hold model data to be used for inference.
* `gpu_introspection.hpp`: Provides constants and utilities to evaluate
  GPU capabilities for optimized performance.
* `infer.hpp`: Contains wrapper code for performing inference on a `forest`
  object (either on CPU or GPU). This wrapper takes data that has been
  extracted from the `forest_model` object if necessary to control details
  of forest evaluation.
* `node.hpp`: Provides template for an individual node of a tree.
* `postprocessor.hpp`: Provides device-agnostic code for postprocessing
  the output of model leaves.
* `specialization_types.hpp`: Defines all specializations that are used to
  construct instantiations of the `decision_forest` template.
* `infer_kernel/`: This directory contains device-specific code that
  determines how `evaluate_tree` calls will be performed in parallel.
* `specializations/`: Because there is a large matrix of
  specializations for `decision_forest`, it would be tedious and
  error-prone to list out all the implementations in source files.
  Furthermore, because these templates are complex we wish to avoid
  recompiling them unnecessarily. Therefore, this directory contains headers
  with macros for declaring the necessary implementations in source files and
  declaring the corresponding templates as `extern` elsewhere. Because
  these specializations need to be explicitly declared, this must be
  implemented as a macro.

### Source Files
The FIL source files contain no implementation details. They
merely use the macros defined in
`include/cuml/fil/detail/specializations` to indicate the template
instantiations that must be compiled. These are broken up into an arbitrary
number of source files. To improve build parallelization, they could be broken
up further, or to reduce the number of source files, they could be
consolidated.
