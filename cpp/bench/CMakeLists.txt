#=============================================================================
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

##############################################################################
# - build cuml bench executable ----------------------------------------------

if(BUILD_CUML_BENCH)
  # (please keep the filenames in alphabetical order)
  add_executable(${CUML_CPP_BENCH_TARGET}
    sg/arima_loglikelihood.cu
    sg/dbscan.cu
    sg/kmeans.cu
    sg/linkage.cu
    sg/main.cpp
    sg/rf_classifier.cu
    # FIXME: RF Regressor is having an issue where the tests now seem to take
    # forever to finish, as opposed to the classifier counterparts!
    # sg/rf_regressor.cu
    sg/svc.cu
    sg/svr.cu
    sg/umap.cu
    sg/fil.cu
  )
  if (CUML_ENABLE_GPU)
    target_compile_definitions(${CUML_CPP_BENCH_TARGET} PUBLIC CUML_ENABLE_GPU)
  endif()

  target_compile_options(${CUML_CPP_BENCH_TARGET}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_link_libraries(${CUML_CPP_BENCH_TARGET}
    PUBLIC
      cuml::${CUML_CPP_TARGET}
      benchmark::benchmark
      ${TREELITE_LIBS}
      ${CUVS_LIB}
      raft::raft
  )

  target_include_directories(${CUML_CPP_BENCH_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
  )

  set_target_properties(
    ${CUML_CPP_BENCH_TARGET}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD                      17
               CXX_STANDARD_REQUIRED             ON
               CUDA_STANDARD                     17
               CUDA_STANDARD_REQUIRED            ON
  )

  install(
    TARGETS ${CUML_CPP_BENCH_TARGET}
    COMPONENT testing
    DESTINATION bin/benchmarks/libcuml
    EXCLUDE_FROM_ALL
  )
endif()
