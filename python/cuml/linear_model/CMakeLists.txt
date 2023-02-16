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
    base.pyx)

add_module_gpu_default("linear_regression.pyx" ${linear_model_algo} ${linearregression_algo})
add_module_gpu_default("elastic_net.pyx" ${elastic_net_algo} ${linearregression_algo})
add_module_gpu_default("logistic_regression.pyx" ${logistic_regression_algo} ${linearregression_algo})
add_module_gpu_default("mbsgd_classifier.pyx" ${mbsgd_classifier_algo} ${linearregression_algo})
add_module_gpu_default("mbsgd_regressor.pyx" ${mbsgd_regressor_algo} ${linearregression_algo})
add_module_gpu_default("ridge.pyx" ${ridge_algo} ${linearregression_algo})

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
  LINKED_LIBRARIES "${cuml_mg_libraries}"
  MODULE_PREFIX linear_model_
  ASSOCIATED_TARGETS cuml
)
