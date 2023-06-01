/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "mgrp_accessor.cuh"

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace CorePoints {

template <typename Index_ = int>
void multi_groups_compute(const raft::handle_t& handle,
                          const Metadata::VertexDegAccessor<Index_, Index_>& vd_ac,
                          Metadata::CorePointAccessor<bool, Index_>& corepts_ac,
                          const Index_* min_pts)
{
  Index_ n_groups       = corepts_ac.n_groups;
  Index_ n_samples      = corepts_ac.n_points;
  const Index_* vd_base = vd_ac.vd;
  bool* mask            = corepts_ac.core_pts;
  const Index_* offset  = corepts_ac.offset_mask;

  auto mask_view = raft::make_device_vector_view(mask, n_samples);
  raft::linalg::map_offset(handle, mask_view, [=] __device__(Index_ idx) {
    return vd_base[idx] >= *(min_pts + offset[idx]);
  });
  return;
}

}  // namespace CorePoints
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML