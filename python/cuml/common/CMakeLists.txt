# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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


set(cython_sources
    base.pyx
    cuda.pyx
    handle.pyx
    logger.pyx
    pointer_utils.pyx
)

set(linked_libraries cuml::${CUML_CPP_TARGET})

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       opg_data_utils_mg.pyx
  )

  list(APPEND linked_libraries
       cumlprims_mg::cumlprims_mg)
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  MODULE_PREFIX common_
)

foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cpp")
endforeach()

# todo: ml_cuda_utils.h should be in the include folder of cuML or the functionality
# moved to another file, pointer_utils.pyx needs it
target_include_directories(pointer_utils PRIVATE "../../../cpp/src/")
