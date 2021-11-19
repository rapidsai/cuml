#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

set(COMMON_TEST_LINK_LIBRARIES
  CUDA::cublas
  CUDA::curand
  CUDA::cusolver
  CUDA::cudart
  CUDA::cusparse
  CUDA::cufft
  rmm::rmm
  raft::raft
  $<$<BOOL:${LINK_FAISS}>:FAISS::FAISS>
  GTest::gtest
  GTest::gtest_main
  OpenMP::OpenMP_CXX
  Threads::Threads
  $<IF:$<BOOL:${Treelite_ADDED}>,treelite::treelite_static,treelite::treelite>
  $<IF:$<BOOL:${Treelite_ADDED}>,treelite::treelite_runtime_static,treelite::treelite_runtime>
  $<TARGET_NAME_IF_EXISTS:conda_env>
)

##############################################################################
# - build ml_test executable -------------------------------------------------

if(BUILD_CUML_TESTS)
  # (please keep the filenames in alphabetical order)

  # Separable executable for each test targetted for 21.12, meanwhile
  # we can just separate FIL here, full algorithm support in progress
  add_executable(${CUML_CPP_TEST_TARGET}
    sg/fil_child_index_test.cu
    sg/fil_test.cu)

  if(CUML_CPP_ALGORITHMS STREQUAL "ALL")
    target_sources(${CUML_CPP_TEST_TARGET}
      PRIVATE
        sg/cd_test.cu
        sg/dbscan_test.cu
        sg/fnv_hash_test.cpp
        sg/genetic/node_test.cpp
        sg/genetic/param_test.cu
        sg/genetic/program_test.cu
        sg/genetic/evolution_test.cu
        sg/hdbscan_test.cu
        sg/holtwinters_test.cu
        sg/kmeans_test.cu
        sg/knn_test.cu
        sg/lars_test.cu
        sg/linear_svm_test.cu
        sg/linkage_test.cu
        sg/logger.cpp
        sg/multi_sum_test.cu
        sg/nvtx_test.cpp
        sg/ols.cu
        sg/pca_test.cu
        sg/quasi_newton.cu
        sg/rf_test.cu
        sg/ridge.cu
        sg/rproj_test.cu
        sg/sgd.cu
        sg/shap_kernel.cu
        sg/svc_test.cu
        sg/trustworthiness_test.cu
        sg/tsne_test.cu
        sg/tsvd_test.cu
        sg/umap_parametrizable_test.cu
        $<$<BOOL:${BUILD_CUML_C_LIBRARY}>:sg/handle_test.cu>
      )
  endif()

  target_compile_options(${CUML_CPP_TEST_TARGET}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_include_directories(${CUML_CPP_TEST_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prims>
  )

  target_link_libraries(${CUML_CPP_TEST_TARGET}
    PRIVATE
        ${CUML_CPP_TARGET}
        $<$<BOOL:${BUILD_CUML_C_LIBRARY}>:${CUML_C_TARGET}>
        ${COMMON_TEST_LINK_LIBRARIES}
    )

endif()

#############################################################################
# - build test_ml_mg executable ----------------------------------------------

if(BUILD_CUML_MG_TESTS)

  if(MPI_CXX_FOUND)
    # (please keep the filenames in alphabetical order)
    add_executable(${CUML_MG_TEST_TARGET}
      mg/knn.cu
      mg/knn_classify.cu
      mg/knn_regress.cu
      mg/main.cu
      mg/pca.cu)

    target_compile_options(${CUML_MG_TEST_TARGET}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
    )

    target_include_directories(${CUML_MG_TEST_TARGET}
      PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prims>
      )

    target_link_libraries(${CUML_MG_TEST_TARGET}
      ${CUML_CPP_TARGET}
      ${COMMON_TEST_LINK_LIBRARIES}
      NCCL::NCCL
      ${MPI_CXX_LIBRARIES}
      cumlprims_mg::cumlprims_mg
    )

  else(MPI_CXX_FOUND)
   message("OpenMPI not found. Skipping test '${CUML_MG_TEST_TARGET}'")
  endif()
endif()

##############################################################################
# - build prims_test executable ----------------------------------------------

if(BUILD_PRIMS_TESTS)
  # (please keep the filenames in alphabetical order)
  add_executable(${PRIMS_TEST_TARGET}
    prims/add_sub_dev_scalar.cu
    prims/adjusted_rand_index.cu
    prims/batched/csr.cu
    prims/batched/gemv.cu
    prims/batched/information_criterion.cu
    prims/batched/make_symm.cu
    prims/batched/matrix.cu
    prims/cache.cu
    prims/columnSort.cu
    prims/completeness_score.cu
    prims/contingencyMatrix.cu
    prims/cov.cu
    prims/decoupled_lookback.cu
    prims/device_utils.cu
    prims/dispersion.cu
    prims/eltwise2d.cu
    prims/entropy.cu
    prims/epsilon_neighborhood.cu
    prims/fast_int_div.cu
    prims/fillna.cu
    prims/gather.cu
    prims/gram.cu
    prims/grid_sync.cu
    prims/hinge.cu
    prims/histogram.cu
    prims/homogeneity_score.cu
    prims/jones_transform.cu
    prims/kl_divergence.cu
    prims/knn_classify.cu
    prims/knn_regression.cu
    prims/kselection.cu
    prims/label.cu
    prims/linalg_block.cu
    prims/linearReg.cu
    prims/log.cu
    prims/logisticReg.cu
    prims/make_arima.cu
    prims/make_blobs.cu
    prims/make_regression.cu
    prims/merge_labels.cu
    prims/minmax.cu
    prims/mvg.cu
    prims/mutual_info_score.cu
    prims/penalty.cu
    prims/permute.cu
    prims/power.cu
    prims/rand_index.cu
    prims/reduce_cols_by_key.cu
    prims/reduce_rows_by_key.cu
    prims/reverse.cu
    prims/rsvd.cu
    prims/score.cu
    prims/seive.cu
    prims/sigmoid.cu
    prims/silhouette_score.cu
    prims/sqrt.cu
    prims/ternary_op.cu
    prims/trustworthiness.cu
    prims/v_measure.cu
    prims/weighted_mean.cu)

  target_compile_options(${PRIMS_TEST_TARGET}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_include_directories(${PRIMS_TEST_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src_prims>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prims>
  )

  target_link_libraries(${PRIMS_TEST_TARGET}
    PRIVATE
      ${CUML_CPP_TARGET}
      ${COMMON_TEST_LINK_LIBRARIES}
  )

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
