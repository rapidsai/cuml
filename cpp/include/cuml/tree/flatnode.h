/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
/* sparse node same tree node in Decsion Tree.
* This however used an index instead of pointer to left child
* Right child index is left_child_id + 1
*/
template <class T, class L>
struct SparseTreeNode {
  L prediction;
  int colid = -1;
  T quesval;
  T best_metric_val;
  int left_child_id = -1;
};
