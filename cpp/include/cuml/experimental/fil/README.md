# Forest Inference Library (FIL)
RAPIDS Forest Inference Library (FIL) provides accelerated inference for
tree-based machine learning models. Unlike packages like XGBoost,
LightGBM, or even Scikit-Learn/cuML's random forest implementations, FIL
cannot be used to _train_ forest models. Instead, its goal is to speed up
inference using forest models trained by all of those packages. 

This directory contains an experimental new implementation of FIL which
provides both CPU and GPU execution. Its GPU implementation also offers
improved performance relative to the existing implementation in many but not all cases. 

For Python usage information and more extensive information on
parameter-tuning and other end-user functionality, check out
TODO(wphicks). This document will focus on the C++ implementation,
offering details on both how to use FIL as a library and how to work with it
as a FIL contributor.

## C++ Usage
All headers required to make use of FIL in another C++ project are
available in the top-level include directory. The `detail` directory
contains implementation details that are not required to use FIL and which
will certainly change over time.

**A NOTE ON THE `kayak` NAMESPACE:** For the first iteration of this FIL
implementation, much of the more general-purpose CPU-GPU interoperable code
has temporarily been put in the `kayak` namespace. As the name suggests,
the intention is that most or all of this functionality will either be moved
to RAFT or that RAFT features will be updated to provide CPU-GPU
compatible versions of the same. The public API includes just a few
references to this namespace, so be prepared to shift these references to the
corresponding RAFT symbols as this version of FIL progresses out of
experimental.

### Importing a model
FIL uses Treelite as a common translation layer for all its input types.
To load a forest model, we first create a Treelite model handle as
follows. Here, we use an XGBoost JSON model as an example, but Treelite has
similar load methods for each of the serialization formats it supports.

```cpp
auto filename = "xgboost.json";
auto tl_model = treelite::frontend::LoadXGBoostModel(filename);
```

We then import the Treelite model into FIL via the
`import_from_treelite_model` function. All arguments except the first are
optional, but we show them all here for illustration.

```cpp
auto stream = cudaStream_t{};
checkCuda(cudaStreamCreate(&stream));

auto fil_model = import_from_treelite_model(
  *tl_model,  // The Treelite model
  128u,  // align_bytes
  false,  // use_double_precision
  kayak::device_type::gpu,  // mem_type
  0,  // device_id
  stream  // CUDA stream
);
```

**align_bytes:** If given a non-zero value, each tree will be padded to a size
that is a multiple of this value by appending additional empty nodes. This
can offer mild performance benefits by increasing the likelihood that memory
reads begin on a cache line boundary. For GPU execution, a value of 128 is
recommended. For most CPUs, a value of 0 is recommended, although using 64 can
occasionally provide benefits.

**use_double_precision**: This argument takes a `std::optional<bool>`. If
`std::nullopt` is used (the default), the *native* precision of the model
serialization format will be used. Otherwise, the model will be evaluated
at double precision if this value is set to `true` or single precision if this
value is set to `false`.

**dev_type**: This argument controls where the model will be executed. If `kayak::device_type::gpu`, then it will be executed on GPU. If `kayak::device_type::cpu`, then it will be executed on CPU.

**device_id**: This integer indicates the ID of the GPU which should be used.
If CPU is being used, this argument is ignored.

**stream**: The CUDA stream which will be used for the actual model import.
If CPU is being used, this argument is ignored. Note that you do *not* need
CUDA headers if you are working with a CPU-only build of FIL. This
argument uses a `kayak::cuda_stream` type which evaluates to a
placeholder type in CPU-only builds. For applications which themselves want to
implement CPU-GPU interoperable builds, the `kayak::cuda_stream` type can be
used directly.


### Inference
The `import_from_treelite_model` function will return a `forest_model` object.
This object has several `predict` methods that can be used to return
inference results for the model. We will describe here the one most likely
to be used by external applications:

```cpp
auto num_rows = std::size_t{1000};
auto num_outputs = fil_model.num_outputs();  // Outputs per row

auto output = static_cast<float*>(nullptr);  // Loaded as single
                                             // precision, so use floats
                                             // for I/O
// Allocate enough space for num_outputs floats per row
cudaMalloc((void**)&output, num_rows * num_outputs * sizeof(float));

// Assuming that input is a float* pointing to data already located on-device

auto handle = kayak::handle_t{};

fil_model.predict(
  handle,
  output,
  input,
  num_rows,
  kayak::device_type::gpu,  // out_mem_type
  kayak::device_type::gpu,  // in_mem_type
  out_mem_type,
  4  // chunk_size
);
```

**handle**: To provide a unified interface on CPU and GPU, we introduce
`kayak::handle_t` as a wrapper for `raft::handle_t`. This is currently just a
placeholder in CPU-only builds, and using it does not require any CUDA
functionality. For GPU-enabled builds, you can construct a
`kayak_handle_t` directly from the `raft::handle_t` you wish to use.

**output**: Pointer to pre-allocated buffer where results should be
written. If the model has been loaded at single precision, this should be a
`float*`. If the model has been loaded at double precision, this should be a
`double*`.

**input**: Pointer to the input data (in C-major order). If the model has been
loaded at single precision, this should be a `float*`. If the model has been
loaded at double precision, this should be a `double*`.

**num_rows**: The number of input rows.

**out_mem_type**: Indicates whether output buffer is on device or host.

**in_mem_type**: Indicates whether input buffer is on device or host.

**chunk_size**: This value has a somewhat different meaning for CPU and GPU,
but it generally indicates the number of rows which are evaluated in a single
iteration of FIL's forest evaluation algorithm. On GPU, any power of 2 from 1 to 32
may be used for this value, and *in general* larger batches benefit from
higher values. Optimizing this value can make an *enormous* difference
in performance and depends on both the model and hardware used to run it. On
CPU, this parameter can take on any value, but powers of 2 between 1 and 512
are recommended. A default value of 64 is generally a safe choice, unless the
batch size is less than 64, in which case a smaller value is recommended. In
general, larger batch sizes benefit from higher chunk size values. This
argument is a `std::optional`, and if `std::nullopt` is passed, a chunk size
will be selected based on heuristics.

## Learning More
While the above usage summary should be enough to get started using FIL in
another C++ application, you can learn more about the details of this
implementation by reading TODO(wphicks).
