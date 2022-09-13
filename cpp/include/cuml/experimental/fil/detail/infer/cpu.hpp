#pragma once
#include <cstddef>
#include <optional>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/cpu_introspection.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/infer_kernel/cpu.hpp>
#include <cuml/experimental/fil/detail/postprocessor.hpp>
#include <cuml/experimental/fil/specializations/infer_macros.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>
namespace herring {
namespace detail {
namespace inference {

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
  index_type class_count,
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
      class_count,
      specified_chunk_size.value_or(hardware_constructive_interference_size),
      hardware_constructive_interference_size,
      vector_output,
      categorical_data
    );
  }
}

HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 0)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 1)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 2)
HERRING_INFER_ALL(extern template, kayak::device_type::cpu, 3)

}
}
}

