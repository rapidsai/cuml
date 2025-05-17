/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/hdbscan.hpp>

void create_kernel_database()
{
  hdbscan_lto_kernels.insert(
    KernelEntry{"condense_hierarchy_kernel",
                "/home/coder/cuml/cpp/build/conda/cuda-12.9/release/CMakeFiles/"
                "hdbscan_kernels_lto.dir/src/hdbscan/detail/kernels/condense.fatbin"});
}
