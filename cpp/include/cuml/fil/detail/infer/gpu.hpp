/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/infer_kind.hpp>

#include <cstddef>
#include <optional>

namespace ML {
namespace fil {
namespace detail {
namespace inference {

/* The CUDA-free header declaration of the GPU infer template */
template <raft_proto::device_type D,
          bool has_categorical_nodes,
          typename forest_t,
          typename vector_output_t    = std::nullptr_t,
          typename categorical_data_t = std::nullptr_t>
std::enable_if_t<D == raft_proto::device_type::gpu, void> infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type class_count,
  vector_output_t vector_output                  = nullptr,
  categorical_data_t categorical_data            = nullptr,
  infer_kind infer_type                          = infer_kind::default_kind,
  std::optional<index_type> specified_chunk_size = std::nullopt,
  raft_proto::device_id<D> device                = raft_proto::device_id<D>{},
  raft_proto::cuda_stream stream                 = raft_proto::cuda_stream{});

}  // namespace inference
}  // namespace detail
}  // namespace fil
}  // namespace ML
