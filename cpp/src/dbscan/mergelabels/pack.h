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
namespace MergeLabels {

template <typename Index_ = int>
struct Pack {
  /** Label array A */
  Index_* labels_a;
  /** Label array B */
  const Index_* labels_b;
  /** Core point mask */
  const bool* mask;
  /** Work buffer */
  Index_* work_buffer;
  /** Flag for the while loop */
  bool* m;
  /** Number of points in the dataset */
  Index_ N;
};

}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
