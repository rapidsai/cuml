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
namespace ML {
namespace experimental {
namespace fil {
namespace detail {

/*
 * Perform inference based on the given forest and input parameters
 *
 * @tparam D The device type (CPU/GPU) used to perform inference
 * @tparam forest_t The type of the forest
 * @param forest The forest to be evaluated
 * @param postproc The postprocessor object used to execute
 * postprocessing
 * @param output Pointer to where the output should be written
 * @param input Pointer to where the input data can be read from
 * @param row_count The number of rows in the input data
 * @param col_count The number of columns in the input data
 * @param output_count The number of outputs per row
 * @param has_categorical_nodes Whether or not any node within the forest has
 * a categorical split
 * @param vector_output Pointer to the beginning of storage for vector
 * outputs of leaves (nullptr for no vector output)
 * @param categorical_data Pointer to external categorical data storage if
 * required
 * @param specified_chunk_size If non-nullopt, the size of "mini-batches"
 * used for distributing work across threads
 * @param device The device on which to execute evaluation
 * @param stream Optionally, the CUDA stream to use
 */
template<kayak::device_type D, typename forest_t>
void infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type output_count,
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
          output_count,
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
          output_count,
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
        output_count,
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
          output_count,
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
          output_count,
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
        output_count,
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
}
}
