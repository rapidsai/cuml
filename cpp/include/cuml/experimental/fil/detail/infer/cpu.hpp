#pragma once
#include <cstddef>
#include <optional>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/cpu_introspection.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/infer_kernel/cpu.hpp>
#include <cuml/experimental/fil/detail/postprocessor.hpp>
#include <cuml/experimental/fil/detail/specializations/infer_macros.hpp>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace inference {

/* A wrapper around the underlying inference kernels to support dispatching to
 * the right kernel
 *
 * This specialization is used for CPU inference and for requests for GPU
 * inference on non-GPU-enabled builds. An exception will be thrown if a
 * request is made for GPU on inference on a non-GPU-enabled build.
 *
 * @tparam D The type of device (CPU/GPU) on which to perform inference.
 * @tparam has_categorical_nodes Whether or not any node in the model has
 * categorical splits.
 * @tparam vector_output_t If non-nullptr_t, the type of vector leaf output
 * @tparam categorical_data_t If non-nullptr_t, the type of non-local
 * categorical data storage
 *
 * @param forest The forest to be used for inference.
 * @param postproc The postprocessor object to be used for postprocessing raw
 * output from the forest.
 * @param row_count The number of rows in the input
 * @param col_count The number of columns per row in the input
 * @param output_count The number of output elements per row
 * @param vector_output If non-nullptr, a pointer to storage for vector leaf
 * outputs
 * @param categorical_data If non-nullptr, a pointer to non-local storage for
 * data on categorical splits.
 * @param specified_chunk_size If non-nullopt, the mini-batch size used for
 * processing rows in a batch. For CPU inference, this essentially determines
 * the granularity of parallelism. A larger chunk size means that a single
 * thread will process more rows for its assigned trees before fetching a
 * new batch of rows. In general, so long as the chunk size remains much
 * smaller than the batch size (minimally less than the batch size divided by
 * the number of available cores), larger batches see improved performance with
 * larger chunk sizes. Unlike for GPU, any positive value is valid (up to
 * hardware constraints), but it is recommended to test powers of 2 from 1
 * (for individual row inference) to 512 (for very large batch
 * inference). A value of 64 is a generally-useful default.
 */
template<
  kayak::device_type D,
  bool has_categorical_nodes,
  typename forest_t,
  typename vector_output_t=std::nullptr_t,
  typename categorical_data_t=std::nullptr_t
>
std::enable_if_t<D==kayak::device_type::cpu || !kayak::GPU_ENABLED, void> infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type output_count,
  vector_output_t vector_output=nullptr,
  categorical_data_t categorical_data=nullptr,
  std::optional<index_type> specified_chunk_size=std::nullopt,
  kayak::device_id<D> device=kayak::device_id<D>{},
  kayak::cuda_stream=kayak::cuda_stream{}
) {
  if constexpr(D==kayak::device_type::gpu) {
    throw kayak::gpu_unsupported("Tried to use GPU inference in CPU-only build");
  } else {
    infer_kernel_cpu<has_categorical_nodes>(
      forest,
      postproc,
      output,
      input,
      row_count,
      col_count,
      output_count,
      specified_chunk_size.value_or(hardware_constructive_interference_size),
      hardware_constructive_interference_size,
      vector_output,
      categorical_data
    );
  }
}

/* This macro is invoked here to declare all standard specializations of this
 * template as extern. This ensures that this (relatively complex) code is
 * compiled as few times as possible. A macro is used because ever
 * specialization must be explicitly declared. The final argument to the macro
 * references the 8 specialization variants compiled in standard cuML FIL. */
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 0)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 1)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 2)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 3)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 4)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 5)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 6)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 7)

}
}
}

}
}
