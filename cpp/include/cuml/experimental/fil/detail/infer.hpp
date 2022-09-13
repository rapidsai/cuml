#pragma once
#include <cstddef>
#include <iostream>
#include <optional>
#include <type_traits>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/infer/cpu.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/fil/detail/infer/gpu.hpp>
#endif
#include <cuml/experimental/fil/detail/postprocessor.hpp>
#include <cuml/experimental/fil/exceptions.hpp>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
namespace herring {
namespace detail {

template<cuml/experimental/kayak::device_type D, typename forest_t>
void infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type class_count,
  bool has_categorical_nodes,
  typename forest_t::io_type* vector_output=nullptr,
  typename forest_t::node_type::index_type* categorical_data=nullptr,
  std::optional<index_type> specified_chunk_size=std::nullopt,
  kayak::device_id<D> device=kayak::device_id<D>{},
  kayak::cuda_stream stream=kayak::cuda_stream{}
) {
  if (vector_output == nullptr) {
    if (categorical_data == nullptr) {
      if (!has_categorical_nodes) {
        inference::infer<D, false, forest_t, std::nullptr_t, std::nullptr_t> (
          forest,
          postproc,
          output,
          input,
          row_count,
          col_count,
          class_count,
          nullptr,
          nullptr,
          specified_chunk_size,
          device,
          stream
        );
      } else {
        inference::infer<D, true, forest_t, std::nullptr_t, std::nullptr_t> (
          forest,
          postproc,
          output,
          input,
          row_count,
          col_count,
          class_count,
          nullptr,
          nullptr,
          specified_chunk_size,
          device,
          stream
        );
      }
    } else {
      inference::infer<D, true, forest_t> (
        forest,
        postproc,
        output,
        input,
        row_count,
        col_count,
        class_count,
        nullptr,
        categorical_data,
        specified_chunk_size,
        device,
        stream
      );
    }
  } else {
    if (categorical_data == nullptr) {
      if (!has_categorical_nodes) {
        inference::infer<D, false, forest_t> (
          forest,
          postproc,
          output,
          input,
          row_count,
          col_count,
          class_count,
          vector_output,
          nullptr,
          specified_chunk_size,
          device,
          stream
        );
      } else {
        inference::infer<D, true, forest_t> (
          forest,
          postproc,
          output,
          input,
          row_count,
          col_count,
          class_count,
          vector_output,
          nullptr,
          specified_chunk_size,
          device,
          stream
        );
      }
    } else {
      inference::infer<D, true, forest_t> (
        forest,
        postproc,
        output,
        input,
        row_count,
        col_count,
        class_count,
        vector_output,
        categorical_data,
        specified_chunk_size,
        device,
        stream
      );
    }
  }
}

}
}
