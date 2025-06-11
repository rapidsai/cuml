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
#pragma once

#include <cuml/fil/constants.hpp>
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/gpu_introspection.hpp>
#include <cuml/fil/detail/infer_kernel/gpu.cuh>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_setter.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>
#include <cuml/fil/detail/specializations/device_initialization_macros.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>

namespace ML {
namespace fil {
namespace detail {
namespace device_initialization {

/* The implementation of the template used to initialize GPU device options
 *
 * On GPU-enabled builds, the GPU specialization of this template ensures that
 * the inference kernels have access to the maximum available dynamic shared
 * memory.
 */
template <typename forest_t, raft_proto::device_type D>
std::enable_if_t<std::conjunction_v<std::bool_constant<raft_proto::GPU_ENABLED>,
                                    std::bool_constant<D == raft_proto::device_type::gpu>>,
                 void>
initialize_device(raft_proto::device_id<D> device)
{
  auto device_context           = raft_proto::device_setter(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // Run solely for side-effect of caching SM count
  get_sm_count(device);
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 1, forest_t, std::nullptr_t, std::nullptr_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 2, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 4, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 8, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 16, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 32, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 1, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 2, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 4, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 8, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 16, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 32, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 1, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 2, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 4, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 8, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 16, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 32, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 1, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 2, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 4, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 8, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 16, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 32, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 1, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 2, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 4, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 8, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 16, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 32, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 1, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 2, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 4, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 8, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 16, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 32, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      1,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      2,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      4,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      8,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      16,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      32,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
}

CUML_FIL_INITIALIZE_DEVICE(extern template, 0)
CUML_FIL_INITIALIZE_DEVICE(extern template, 1)
CUML_FIL_INITIALIZE_DEVICE(extern template, 2)
CUML_FIL_INITIALIZE_DEVICE(extern template, 3)
CUML_FIL_INITIALIZE_DEVICE(extern template, 4)
CUML_FIL_INITIALIZE_DEVICE(extern template, 5)
CUML_FIL_INITIALIZE_DEVICE(extern template, 6)
CUML_FIL_INITIALIZE_DEVICE(extern template, 7)
CUML_FIL_INITIALIZE_DEVICE(extern template, 8)
CUML_FIL_INITIALIZE_DEVICE(extern template, 9)
CUML_FIL_INITIALIZE_DEVICE(extern template, 10)
CUML_FIL_INITIALIZE_DEVICE(extern template, 11)

}  // namespace device_initialization
}  // namespace detail
}  // namespace fil

}  // namespace ML
