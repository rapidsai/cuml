#include <cstddef>
#include <optional>
#include <herring3/detail/forest.hpp>
#include <herring3/detail/index_type.hpp>
#include <herring3/detail/postprocessor.hpp>
#include <kayak/cuda_stream.hpp>
#include <kayak/device_id.hpp>
#include <kayak/device_type.hpp>

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
std::enable_if_t<D==kayak::device_type::gpu, void> infer(
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
  kayak::cuda_stream stream=kayak::cuda_stream{}
);

}
}
}
