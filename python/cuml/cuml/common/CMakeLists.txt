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
add_module_gpu_default("pointer_utils.pyx")

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       opg_data_utils_mg.pyx
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${cuml_mg_libraries}"
  MODULE_PREFIX common_
)

if(${CUML_UNIVERSAL})
# todo: ml_cuda_utils.h should be in the include folder of cuML or the functionality
# moved to another file, pointer_utils.pyx needs it
# https://github.com/rapidsai/cuml/issues/4841
  target_include_directories(common_pointer_utils PRIVATE "../../../../cpp/src/")
endif()
