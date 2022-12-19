#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

function(ConfigureTest)

  set(options OPTIONAL CUMLPRIMS MPI ML_INCLUDE RAFT_DISTRIBUTED)
  set(oneValueArgs PREFIX NAME PATH)
  set(multiValueArgs TARGETS CONFIGURATIONS)

  cmake_parse_arguments(ConfigureTest "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

  set(TEST_NAME ${ConfigureTest_PREFIX}_${ConfigureTest_NAME})
  if(ConfigureTest_PREFIX STREQUAL "PRIMS")
    set(TEST_INSTALL_PATH bin/gtests/libcuml_prims)
  else()
    set(TEST_INSTALL_PATH bin/gtests/libcuml)
  endif()

  add_executable(${TEST_NAME} ${ConfigureTest_PATH})

  target_link_libraries(${TEST_NAME}
  PRIVATE
    ${CUML_CPP_TARGET}
    $<$<BOOL:BUILD_CUML_C_LIBRARY>:${CUML_C_TARGET}>
    CUDA::cublas${_ctk_static_suffix}
    CUDA::curand${_ctk_static_suffix}
    CUDA::cusolver${_ctk_static_suffix}
    CUDA::cudart${_ctk_static_suffix}
    CUDA::cusparse${_ctk_static_suffix}
    $<$<BOOL:${LINK_CUFFT}>:CUDA::cufft${_ctk_static_suffix_cufft}>
    rmm::rmm
    raft::raft
    $<$<BOOL:${CUML_USE_RAFT_NN}>:raft::nn>
    $<$<BOOL:${CUML_USE_RAFT_DIST}>:raft::distance>
    GTest::gtest
    GTest::gtest_main
    ${OpenMP_CXX_LIB_NAMES}
    Threads::Threads
    $<$<BOOL:${ConfigureTest_CUMLPRIMS}>:cumlprims_mg::cumlprims_mg>
    $<$<BOOL:${ConfigureTest_MPI}>:${MPI_CXX_LIBRARIES}>
    $<$<BOOL:${ConfigureTest_RAFT_DISTANCE}>:raft::distributed>
    ${TREELITE_LIBS}
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  target_compile_options(${TEST_NAME}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_include_directories(${TEST_NAME}
    PRIVATE
      $<$<BOOL:${ConfigureTest_ML_INCLUDE}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>>
      $<$<BOOL:${ConfigureTest_ML_INCLUDE}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prims>
  )

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    set_target_properties(
      ${TEST_NAME}
      PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
    )

    install(
      TARGETS ${TEST_NAME}
      COMPONENT testing
      DESTINATION ${TEST_INSTALL_PATH}
      EXCLUDE_FROM_ALL
    )

endfunction()


##############################################################################
# - build ml_test executable -------------------------------------------------

if(BUILD_CUML_TESTS)

  if(all_algo)
    ConfigureTest(PREFIX SG NAME LOGGER_TEST PATH sg/logger.cpp OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR dbscan_algo)
    ConfigureTest(PREFIX SG NAME DBSCAN_TEST PATH sg/dbscan_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR explainer_algo)
    ConfigureTest(PREFIX SG NAME SHAP_KERNEL_TEST PATH sg/shap_kernel.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR fil_algo)
    ConfigureTest(PREFIX SG NAME FIL_CHILD_INDEX_TEST PATH sg/fil_child_index_test.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME FIL_TEST PATH sg/fil_test.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME FNV_HASH_TEST PATH sg/fnv_hash_test.cpp OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME MULTI_SUM_TEST PATH sg/multi_sum_test.cu OPTIONAL ML_INCLUDE)
  endif()

  # todo: organize linear models better
  if(all_algo OR linearregression_algo OR ridge_algo OR lasso_algo OR logisticregression_algo)
    ConfigureTest(PREFIX SG NAME OLS_TEST PATH sg/ols.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME RIDGE_TEST PATH sg/ridge.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR genetic_algo)
    ConfigureTest(PREFIX SG NAME GENETIC_NODE_TEST PATH sg/genetic/node_test.cpp OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME GENETIC_PARAM_TEST PATH sg/genetic/param_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if("${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS_EQUAL "11.2")
      # An HDBSCAN gtest is failing w/ CUDA 11.2 for some reason.
      if(all_algo OR hdbscan_algo)
        ConfigureTest(PREFIX SG NAME HDBSCAN_TEST PATH sg/hdbscan_test.cu OPTIONAL ML_INCLUDE)
      endif()
  endif()


  if(all_algo OR holtwinters_algo)
    ConfigureTest(PREFIX SG NAME HOLTWINTERS_TEST PATH sg/holtwinters_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR knn_algo)
    ConfigureTest(PREFIX SG NAME KNN_TEST PATH sg/knn_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR hierarchicalclustering_algo)
    ConfigureTest(PREFIX SG NAME LINKAGE_TEST PATH sg/linkage_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR metrics_algo)
    ConfigureTest(PREFIX SG NAME TRUSTWORTHINESS_TEST PATH sg/trustworthiness_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR pca_algo)
    ConfigureTest(PREFIX SG NAME PCA_TEST PATH sg/pca_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR randomforest_algo)
    ConfigureTest(PREFIX SG NAME RF_TEST PATH sg/rf_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR randomprojection_algo)
    ConfigureTest(PREFIX SG NAME RPROJ_TEST PATH sg/rproj_test.cu OPTIONAL ML_INCLUDE)
  endif()

  # todo: separate solvers better
  if(all_algo OR solvers_algo)
    ConfigureTest(PREFIX SG NAME CD_TEST PATH sg/cd_test.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME LARS_TEST PATH sg/lars_test.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME QUASI_NEWTON PATH sg/quasi_newton.cu OPTIONAL ML_INCLUDE)
    ConfigureTest(PREFIX SG NAME SGD_TEST PATH sg/sgd.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR svm_algo)
    ConfigureTest(PREFIX SG NAME SVC_TEST PATH sg/svc_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR tsne_algo)
    ConfigureTest(PREFIX SG NAME TSNE_TEST PATH sg/tsne_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR tsvd_algo)
    ConfigureTest(PREFIX SG NAME TSVD_TEST PATH sg/tsvd_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(all_algo OR umap_algo)
    ConfigureTest(PREFIX SG NAME UMAP_PARAMETRIZABLE_TEST PATH sg/umap_parametrizable_test.cu OPTIONAL ML_INCLUDE)
  endif()

  if(BUILD_CUML_C_LIBRARY)
    ConfigureTest(PREFIX SG NAME HANDLE_TEST PATH sg/handle_test.cu OPTIONAL ML_INCLUDE)
  endif()

endif()

#############################################################################
# - build test_ml_mg executable ----------------------------------------------

if(BUILD_CUML_MG_TESTS)

  # This test needs to be rewritten to use the MPI comms, not the std comms, and moved
  # to RAFT: https://github.com/rapidsai/cuml/issues/5058
  #ConfigureTest(PREFIX MG NAME KMEANS_TEST PATH mg/kmeans_test.cu OPTIONAL NCCL CUMLPRIMS ML_INCLUDE)
  if(MPI_CXX_FOUND)
    # (please keep the filenames in alphabetical order)
    ConfigureTest(PREFIX MG NAME KNN_TEST PATH mg/knn.cu OPTIONAL CUMLPRIMS MPI RAFT_DISTRIBUTED ML_INCLUDE)
    ConfigureTest(PREFIX MG NAME KNN_CLASSIFY_TEST PATH mg/knn_classify.cu OPTIONAL CUMLPRIMS MPI RAFT_DISTRIBUTED ML_INCLUDE)
    ConfigureTest(PREFIX MG NAME KNN_REGRESS_TEST PATH mg/knn_regress.cu OPTIONAL CUMLPRIMS MPI RAFT_DISTRIBUTED ML_INCLUDE)
    ConfigureTest(PREFIX MG NAME MAIN_TEST PATH mg/main.cu OPTIONAL CUMLPRIMS MPI RAFT_DISTRIBUTED ML_INCLUDE)
    ConfigureTest(PREFIX MG NAME PCA_TEST PATH mg/pca.cu OPTIONAL CUMLPRIMS MPI RAFT_DISTRIBUTED ML_INCLUDE)
  else(MPI_CXX_FOUND)
   message("OpenMPI not found. Skipping MultiGPU tests '${CUML_MG_TEST_TARGET}'")
  endif()
endif()

##############################################################################
# - build prims_test executable ----------------------------------------------

if(BUILD_PRIMS_TESTS)
  # (please keep the filenames in alphabetical order)
  ConfigureTest(PREFIX PRIMS NAME ADD_SUB_DEV_SCALAR_TEST PATH prims/add_sub_dev_scalar.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_CSR_TEST PATH prims/batched/csr.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_GEMV_TEST PATH prims/batched/gemv.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_MAKE_SYMM_TEST PATH prims/batched/make_symm.cu)
  ConfigureTest(PREFIX PRIMS NAME BATCHED_MATRIX_TEST PATH prims/batched/matrix.cu)
  ConfigureTest(PREFIX PRIMS NAME DECOUPLED_LOOKBACK_TEST PATH prims/decoupled_lookback.cu)
  ConfigureTest(PREFIX PRIMS NAME DEVICE_UTILS_TEST PATH prims/device_utils.cu)
  ConfigureTest(PREFIX PRIMS NAME ELTWISE2D_TEST PATH prims/eltwise2d.cu)
  ConfigureTest(PREFIX PRIMS NAME FAST_INT_DIV_TEST PATH prims/fast_int_div.cu)
  ConfigureTest(PREFIX PRIMS NAME FILLNA_TEST PATH prims/fillna.cu)
  ConfigureTest(PREFIX PRIMS NAME GRID_SYNC_TEST PATH prims/grid_sync.cu)
  ConfigureTest(PREFIX PRIMS NAME HINGE_TEST PATH prims/hinge.cu)
  ConfigureTest(PREFIX PRIMS NAME JONES_TRANSFORM_TEST PATH prims/jones_transform.cu)
  ConfigureTest(PREFIX PRIMS NAME KNN_CLASSIFY_TEST PATH prims/knn_classify.cu)
  ConfigureTest(PREFIX PRIMS NAME KNN_REGRESSION_TEST PATH prims/knn_regression.cu)
  ConfigureTest(PREFIX PRIMS NAME KSELECTION_TEST PATH prims/kselection.cu)
  ConfigureTest(PREFIX PRIMS NAME LINALG_BLOCK_TEST PATH prims/linalg_block.cu)
  ConfigureTest(PREFIX PRIMS NAME LINEARREG_TEST PATH prims/linearReg.cu)
  ConfigureTest(PREFIX PRIMS NAME LOG_TEST PATH prims/log.cu)
  ConfigureTest(PREFIX PRIMS NAME LOGISTICREG_TEST PATH prims/logisticReg.cu)
  ConfigureTest(PREFIX PRIMS NAME MAKE_ARIMA_TEST PATH prims/make_arima.cu)
  ConfigureTest(PREFIX PRIMS NAME PENALTY_TEST PATH prims/penalty.cu)
  ConfigureTest(PREFIX PRIMS NAME REVERSE_TEST PATH prims/reverse.cu)
  ConfigureTest(PREFIX PRIMS NAME SIGMOID_TEST PATH prims/sigmoid.cu)

endif()

##############################################################################
# - build C-API test library -------------------------------------------------

if(BUILD_CUML_C_LIBRARY)

  enable_language(C)

  add_library(${CUML_C_TEST_TARGET} SHARED
    c_api/dbscan_api_test.c
    c_api/glm_api_test.c
    c_api/holtwinters_api_test.c
    c_api/knn_api_test.c
    c_api/svm_api_test.c
  )

  target_link_libraries(${CUML_C_TEST_TARGET} PUBLIC ${CUML_C_TARGET})

endif()
