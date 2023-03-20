/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cstddef>
#include <type_traits>
#include <variant>
#include <cuml/experimental/fil/decision_forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/buffer.hpp>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>
#include <cuml/experimental/fil/detail/raft_proto/handle.hpp>

namespace ML {
namespace experimental {
namespace fil {

/**
 * A model used for performing inference with FIL
 *
 * This struct is a wrapper for all variants of decision_forest supported by a
 * standard FIL build.
 */
struct forest_model {
  /** Wrap a decision_forest in a full forest_model object */
  forest_model(
    decision_forest_variant&& forest = decision_forest_variant{}
  ) : decision_forest_{forest} {}

  /** The number of features per row expected by the model */
  auto num_features() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.num_features();
    }, decision_forest_);
  }

  /** The number of outputs per row generated by the model */
  auto num_outputs() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.num_outputs();
    }, decision_forest_);
  }

  /** The type of memory (device/host) where the model is stored */
  auto memory_type() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.memory_type();
    }, decision_forest_);
  }

  /** The ID of the device on which this model is loaded */
  auto device_index() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.device_index();
    }, decision_forest_);
  }

  /** Whether or not model is loaded at double precision */
  auto is_double_precision() {
    return std::visit([](auto&& concrete_forest) {
      return std::is_same_v<
        typename std::remove_reference_t<decltype(concrete_forest)>::io_type, double
      >;
    }, decision_forest_);
  }

  /**
   * Perform inference on given input
   *
   * @param[out] output The buffer where model output should be stored.
   * This must be of size at least ROWS x num_outputs().
   * @param[in] input The buffer containing input data.
   * @param[in] stream A raft_proto::cuda_stream, which (on GPU-enabled builds) is
   * a transparent wrapper for the cudaStream_t or (on CPU-only builds) a
   * CUDA-free placeholder object.
   * @param[in] specified_chunk_size: Specifies the mini-batch size for
   * processing. This has different meanings on CPU and GPU, but on GPU it
   * corresponds to the number of rows evaluated per inference iteration
   * on a single block. It can take on any power of 2 from 1 to 32, and
   * runtime performance is quite sensitive to the value chosen. In general,
   * larger batches benefit from higher values, but it is hard to predict the
   * optimal value a priori. If omitted, a heuristic will be used to select a
   * reasonable value. On CPU, this argument can generally just be omitted.
   */
  template <typename io_t>
  void predict(
    raft_proto::buffer<io_t>& output,
    raft_proto::buffer<io_t> const& input,
    raft_proto::cuda_stream stream = raft_proto::cuda_stream{},
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    std::visit([this, &output, &input, &stream, &specified_chunk_size](auto&& concrete_forest) {
      if constexpr(std::is_same_v<typename std::remove_reference_t<decltype(concrete_forest)>::io_type, io_t>) {
        concrete_forest.predict(output, input, stream, specified_chunk_size);
      } else {
        throw type_error("Input type does not match model_type");
      }
    }, decision_forest_);
  }

  /**
   * Perform inference on given input
   *
   * @param[in] handle The raft_proto::handle_t (wrapper for raft::handle_t
   * on GPU) which will be used to provide streams for evaluation.
   * @param[out] output The buffer where model output should be stored. If
   * this buffer is on host while the model is on device or vice versa,
   * work will be distributed across available streams to copy the data back
   * to this output location. This must be of size at least ROWS x num_outputs().
   * @param[in] input The buffer containing input data. If
   * this buffer is on host while the model is on device or vice versa,
   * work will be distributed across available streams to copy the input data
   * to the appropriate location and perform inference.
   * @param[in] specified_chunk_size: Specifies the mini-batch size for
   * processing. This has different meanings on CPU and GPU, but on GPU it
   * corresponds to the number of rows evaluated per inference iteration
   * on a single block. It can take on any power of 2 from 1 to 32, and
   * runtime performance is quite sensitive to the value chosen. In general,
   * larger batches benefit from higher values, but it is hard to predict the
   * optimal value a priori. If omitted, a heuristic will be used to select a
   * reasonable value. On CPU, this argument can generally just be omitted.
   */
  template <typename io_t>
  void predict(
    raft_proto::handle_t const& handle,
    raft_proto::buffer<io_t>& output,
    raft_proto::buffer<io_t> const& input,
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    std::visit([this, &handle, &output, &input, &specified_chunk_size](auto&& concrete_forest) {
      using model_io_t = typename std::remove_reference_t<decltype(concrete_forest)>::io_type;
      if constexpr(std::is_same_v<model_io_t, io_t>) {
        if (output.memory_type() == memory_type() && input.memory_type() == memory_type()) {
          concrete_forest.predict(
            output,
            input,
            handle.get_next_usable_stream(),
            specified_chunk_size
          );
        } else {
          auto constexpr static const MIN_CHUNKS_PER_PARTITION = std::size_t{64};
          auto constexpr static const MAX_CHUNK_SIZE = std::size_t{64};

          auto row_count = input.size() / num_features();
          auto partition_size = std::max(
            raft_proto::ceildiv(row_count, handle.get_usable_stream_count()),
            specified_chunk_size.value_or(MAX_CHUNK_SIZE) * MIN_CHUNKS_PER_PARTITION
          );
          auto partition_count = raft_proto::ceildiv(row_count, partition_size);
          for (auto i = std::size_t{}; i < partition_count; ++i) {
            auto stream = handle.get_next_usable_stream();
            auto rows_in_this_partition = std::min(partition_size, row_count - i * partition_size);
            auto partition_in = raft_proto::buffer<io_t>{};
            if (input.memory_type() != memory_type()) {
              partition_in = raft_proto::buffer<io_t>{
                rows_in_this_partition * num_features(),
                memory_type()
              };
              raft_proto::copy<raft_proto::DEBUG_ENABLED>(
                partition_in,
                input,
                0,
                i * partition_size * num_features(),
                partition_in.size(),
                stream
              );
            } else {
              partition_in = raft_proto::buffer<io_t>{
                input.data() + i * partition_size * num_features(),
                rows_in_this_partition * num_features(),
                memory_type()
              };
            }
            auto partition_out = raft_proto::buffer<io_t>{};
            if (output.memory_type() != memory_type()) {
              partition_out = raft_proto::buffer<io_t>{
                rows_in_this_partition * num_outputs(),
                memory_type()
              };
            } else {
              partition_out = raft_proto::buffer<io_t>{
                output.data() + i * partition_size * num_outputs(),
                rows_in_this_partition * num_outputs(),
                memory_type()
              };
            }
            concrete_forest.predict(
              partition_out,
              partition_in,
              stream,
              specified_chunk_size
            );
            if (output.memory_type() != memory_type()) {
              raft_proto::copy<raft_proto::DEBUG_ENABLED>(
                output,
                partition_out,
                i * partition_size * num_outputs(),
                0,
                partition_out.size(),
                stream
              );
            }
          }
        }
      } else {
        throw type_error("Input type does not match model_type");
      }
    }, decision_forest_);
  }

  /**
   * Perform inference on given input
   *
   * @param[in] handle The raft_proto::handle_t (wrapper for raft::handle_t
   * on GPU) which will be used to provide streams for evaluation.
   * @param[out] output Pointer to the memory location where output should end
   * up
   * @param[in] input Pointer to the input data
   * @param[in] num_rows Number of rows in input
   * @param[in] out_mem_type The memory type (device/host) of the output
   * buffer
   * @param[in] in_mem_type The memory type (device/host) of the input buffer
   * @param[in] specified_chunk_size: Specifies the mini-batch size for
   * processing. This has different meanings on CPU and GPU, but on GPU it
   * corresponds to the number of rows evaluated per inference iteration
   * on a single block. It can take on any power of 2 from 1 to 32, and
   * runtime performance is quite sensitive to the value chosen. In general,
   * larger batches benefit from higher values, but it is hard to predict the
   * optimal value a priori. If omitted, a heuristic will be used to select a
   * reasonable value. On CPU, this argument can generally just be omitted.
   */
  template <typename io_t>
  void predict(
    raft_proto::handle_t const& handle,
    io_t* output,
    io_t* input,
    std::size_t num_rows,
    raft_proto::device_type out_mem_type,
    raft_proto::device_type in_mem_type,
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    // TODO(wphicks): Make sure buffer lands on same device as model
    auto out_buffer = raft_proto::buffer{
      output,
      num_rows * num_outputs(),
      out_mem_type
    };
    auto in_buffer = raft_proto::buffer{
      input,
      num_rows * num_features(),
      in_mem_type
    };
    predict(handle, out_buffer, in_buffer, specified_chunk_size);
  }

 private:
  decision_forest_variant decision_forest_;
};

}
}
}