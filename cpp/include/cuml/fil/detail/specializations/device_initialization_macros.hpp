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
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/specializations/forest_macros.hpp>
/* Declare device initialization function for the types specified by the given
 * variant index */
#define CUML_FIL_INITIALIZE_DEVICE(template_type, variant_index)                     \
  template_type void                                                                 \
    initialize_device<CUML_FIL_FOREST(variant_index), raft_proto::device_type::gpu>( \
      raft_proto::device_id<raft_proto::device_type::gpu>);
