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
#include <type_traits>
#include <cuml/experimental/fil/detail/raft_proto/device_id.hpp>
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace device_initialization {

/* Specialization for any initialization required for CPUs
 *
 * This specialization will also be used for non-GPU-enabled builds
 * (as a GPU no-op).
 */
template<typename forest_t, raft_proto::device_type D>
std::enable_if_t<D == raft_proto::device_type::cpu, void> initialize_device(raft_proto::device_id<D> device) {}

/* Note(wphicks): In the above template, it should be possible to add
 * `|| ! raft_proto::GPU_ENABLED` to the enable_if clause. This works in gcc 9
 * but not gcc 11. As a workaround, we use the following ifdef. If this is
 * corrected in a later gcc version, we can remove the following and just use
 * the above template. Alternatively, if we see some way in which the above is
 * actually an abuse of SFINAE that was accidentally permitted by gcc 9, the
 * root cause should be corrected. */
#ifndef CUML_ENABLE_GPU
template<typename forest_t, raft_proto::device_type D>
std::enable_if_t<D == raft_proto::device_type::gpu, void> initialize_device(raft_proto::device_id<D> device) {}
#endif

}
}
}
}
}
