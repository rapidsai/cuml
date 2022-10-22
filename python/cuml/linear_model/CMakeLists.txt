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


# set(cython_sources
#     base.pyx
#     elastic_net.pyx
#     linear_regression.pyx
#     logistic_regression.pyx
#     mbsgd_classifier.pyx
#     mbsgd_regressor.pyx
#     ridge.pyx
# )

function(add_cython_module FILES_LIST GROUP_FLAG INDIVIDUAL_FLAG FILENAME)
    if(GROUP_FLAG OR INDIVIDUAL_FLAG)
      list(APPEND FILES_LIST
           FILENAME)
      set (FILES_LIST  ${FILES_LIST} PARENT_SCOPE)
    endif(GROUP_FLAG OR INDIVIDUAL_FLAG)
endfunction()


set(cython_sources
    base.pyx)

add_cython_module(cython_sources linear_model_algo linearregression_algo "linear_regression.pyx")

set(cython_sources
    base.pyx
    elastic_net.pyx
    logistic_regression.pyx
    mbsgd_classifier.pyx
    mbsgd_regressor.pyx
    ridge.pyx
)

message("@@@DEBUG@@@ CUML_PY: cython_sources: '${cython_sources}'.")

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       base_mg.pyx
       linear_regression_mg.pyx
       ridge_mg.pyx
  )

endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES $<IF:$<BOOL:${SINGLE_GPU}>,${cuml_sg_libraries},${cuml_mg_libraries}>
  MODULE_PREFIX linear_model_
)

foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../library")
endforeach()
