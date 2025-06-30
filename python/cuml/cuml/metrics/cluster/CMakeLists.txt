# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================



set(cython_sources "")
add_module_gpu_default("adjusted_rand_index.pyx" ${adjusted_rand_index_algo} ${metrics_algo})
add_module_gpu_default("completeness_score.pyx" ${completeness_score_algo} ${metrics_algo})
add_module_gpu_default("entropy.pyx" ${entropy_algo} ${metrics_algo})
add_module_gpu_default("homogeneity_score.pyx" ${homogeneity_score_algo} ${metrics_algo})
add_module_gpu_default("mutual_info_score.pyx" ${mutual_info_score_algo} ${metrics_algo})
add_module_gpu_default("silhouette_score.pyx" ${silhouette_score_algo} ${metrics_algo})
add_module_gpu_default("v_measure.pyx" ${v_measure_algo} ${metrics_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}"
)
