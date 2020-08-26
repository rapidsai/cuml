/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "rmmAllocatorAdapter.hpp"

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>

namespace ML {

namespace {
// simple helpers for creating RMM MRs
inline auto make_cuda() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

inline auto make_pool() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda());
}
}  // namespace

/**
 * @brief Implemententation of ML::deviceAllocator using the RMM pool
 *
 * @todo rmmPoolAllocatorAdapter currently only uses the default ctor of the
 *       underlying device_memory_resource.
 */
class rmmPoolAllocatorAdapter : public rmmAllocatorAdapter {
 public:
  rmmPoolAllocatorAdapter() : mr_(make_pool()) {
    prev_mr_ = rmm::mr::set_current_device_resource(mr_.get());
  }

  ~rmmPoolAllocatorAdapter() {
    // restore the previous memory resource when this object goes out-of-scope
    rmm::mr::set_current_device_resource(prev_mr_);
  }

 private:
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
  rmm::mr::device_memory_resource* prev_mr_;
};

}  // end namespace ML
