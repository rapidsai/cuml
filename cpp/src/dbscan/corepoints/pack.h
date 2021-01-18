/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace ML {
namespace Dbscan {
namespace CorePoints {

template <typename Index_>
struct Pack {
  /** vertex degree array */
  const Index_ *vd;
  /** core point mask */
  bool *mask;
  /** core points criterion */
  Index_ min_pts;
};

}  // namespace CorePoints
}  // namespace Dbscan
}  // namespace ML