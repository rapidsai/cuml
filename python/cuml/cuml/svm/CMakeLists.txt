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
add_module_gpu_default("linear.pyx" ${linear_svm_algo} ${svm_algo})
add_module_gpu_default("svm_base.pyx" ${linear_svm_algo} ${svc_algo} ${svr_algo} ${svm_algo})

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_sg_libraries}"
  MODULE_PREFIX svm_
)
