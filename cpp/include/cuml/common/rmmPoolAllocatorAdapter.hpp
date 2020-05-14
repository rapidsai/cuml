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

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>
#include "rmmAllocatorAdapter.hpp"
#include <rmm/mr/device/cnmem_memory_resource.hpp>

namespace ML {

/**
 * @brief Implemententation of ML::deviceAllocator using the RMM pool
 *
 * @todo rmmPoolAllocatorAdapter currently only uses the default ctor of the
 *       underlying pool allocator (ie cnmem).
 */
class rmmPoolAllocatorAdapter : public rmmAllocatorAdapter {
 public:
  rmmPoolAllocatorAdapter() : cnmem_mr_() {
    prev_mr_ = rmm::mr::set_default_resource(&cnmem_mr_);
  }

  ~rmmPoolAllocatorAdapter() {
    // restore the previous memory resource when this object goes out-of-scope
    rmm::mr::set_default_resource(prev_mr_);
  }

 private:
  rmm::mr::cnmem_memory_resource cnmem_mr_;
  rmm::mr::device_memory_resource* prev_mr_;
};

}  // end namespace ML
