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


set(cython_sources "")
add_module_gpu_default("cd.pyx" ${cd_algo} ${elastic_net_algo} ${lasso_algo} ${solvers_algo})
add_module_gpu_default("qn.pyx" ${qn_algo} ${elastic_net_algo} ${logisticregression_algo} ${solvers_algo})
# Note: add sgd regressor and classifier flags to SGD if CPU support is added
add_module_gpu_default("sgd.pyx" ${sgd_algo} ${solvers_algo})

if(NOT SINGLEGPU)
  list(APPEND cython_sources
       cd_mg.pyx
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES $<IF:$<BOOL:${SINGLE_GPU}>,${cuml_sg_libraries},${cuml_mg_libraries}>
  MODULE_PREFIX solvers_
  ASSOCIATED_TARGETS cuml
)
