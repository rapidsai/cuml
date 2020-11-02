#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

set(CUML_TEST_INCLUDE_DIRS
  ${CUML_INCLUDE_DIRECTORIES}
  ${GTEST_INCLUDE_DIRS}/include
  ${TREELITE_DIR}/runtime/native/include)

set(CUML_TEST_LINK_LIBRARIES
  ${CUML_CPP_TARGET}
  FAISS::FAISS
  GTest::GTest
  GTest::Main
  treelite::treelite
  treelite::treelite_runtime
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  )

set(PRIMS_TEST_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY})

##############################################################################
# - build ml_test executable -------------------------------------------------

if(BUILD_CUML_TESTS)
  # (please keep the filenames in alphabetical order)
  add_executable(ml
    sg/cd_test.cu
    sg/dbscan_test.cu
    sg/decisiontree_batchedlevel_algo.cu
    sg/fil_test.cu
    sg/handle_test.cu
    sg/holtwinters_test.cu
    sg/kmeans_test.cu
    sg/knn_test.cu
    sg/logger.cpp
    sg/nvtx_test.cpp
    sg/ols.cu
    sg/pca_test.cu
    sg/quasi_newton.cu
    sg/rf_accuracy_test.cu
    sg/rf_batched_classification_test.cu
    sg/rf_batched_regression_test.cu
    sg/rf_depth_test.cu
    sg/rf_test.cu
    sg/rf_treelite_test.cu
    sg/ridge.cu
    sg/rproj_test.cu
    sg/sgd.cu
    sg/svc_test.cu
    sg/trustworthiness_test.cu
    sg/tsne_test.cu
    sg/tsvd_test.cu
    sg/umap_parametrizable_test.cu)

  add_dependencies(ml cutlass)

  target_include_directories(ml PRIVATE ${CUML_TEST_INCLUDE_DIRS})

  target_link_libraries(ml ${CUML_TEST_LINK_LIBRARIES})

endif(BUILD_CUML_TESTS)

##############################################################################
# - build test_ml_mg executable ----------------------------------------------

if(BUILD_CUML_MG_TESTS)

  if(MPI_CXX_FOUND)
    # (please keep the filenames in alphabetical order)
    add_executable(ml_mg
      mg/knn.cu
      mg/knn_classify.cu
      mg/knn_regress.cu
      mg/main.cu
      mg/pca.cu)

    set(CUML_TEST_INCLUDE_DIRS
        ${CUML_TEST_INCLUDE_DIRS}
        NCCL::NCCL
        ${MPI_CXX_INCLUDE_PATH}
        ${cumlprims_mg_INCLUDE_DIRS})

    target_include_directories(ml_mg PUBLIC ${CUML_TEST_INCLUDE_DIRS})

    set(CUML_TEST_LINK_LIBRARIES
    	${CUML_TEST_LINK_LIBRARIES}
    	NCCL::NCCL
    	${MPI_CXX_LIBRARIES}
    	cumlprims_mg::cumlprims_mg)

    target_link_libraries(ml_mg ${CUML_TEST_LINK_LIBRARIES})

  else(MPI_CXX_FOUND)
   message("OpenMPI not found. Skipping 'ml_mg'")
  endif(MPI_CXX_FOUND)
endif(BUILD_CUML_MG_TESTS)

##############################################################################
# - build prims_test executable ----------------------------------------------

if(BUILD_PRIMS_TESTS)
  # (please keep the filenames in alphabetical order)
  add_executable(prims
    prims/add_sub_dev_scalar.cu
    prims/adjustedRandIndex.cu
    prims/batched/csr.cu
    prims/batched/gemv.cu
    prims/batched/information_criterion.cu
    prims/batched/make_symm.cu
    prims/batched/matrix.cu
    prims/cache.cu
    prims/columnSort.cu
    prims/completenessScore.cu
    prims/contingencyMatrix.cu
    prims/coo.cu
    prims/cov.cu
    prims/csr.cu
    prims/cutlass_gemm.cu
    prims/decoupled_lookback.cu
    prims/device_utils.cu
    prims/dispersion.cu
    prims/dist_adj.cu
    prims/dist_cos.cu
    prims/dist_euc_exp.cu
    prims/dist_euc_unexp.cu
    prims/dist_l1.cu
    prims/eltwise2d.cu
    prims/entropy.cu
    prims/epsilon_neighborhood.cu
    prims/fast_int_div.cu
    prims/fused_l2_nn.cu
    prims/gather.cu
    prims/gram.cu
    prims/grid_sync.cu
    prims/hinge.cu
    prims/histogram.cu
    prims/homogeneityScore.cu
    prims/host_buffer.cu
    prims/jones_transform.cu
    prims/klDivergence.cu
    prims/knn_classify.cu
    prims/knn_regression.cu
    prims/knn.cu
    prims/kselection.cu
    prims/label.cu
    prims/linearReg.cu
    prims/log.cu
    prims/logisticReg.cu
    prims/make_arima.cu
    prims/make_blobs.cu
    prims/make_regression.cu
    prims/minmax.cu
    prims/mvg.cu
    prims/mutualInfoScore.cu
    prims/penalty.cu
    prims/permute.cu
    prims/power.cu
    prims/randIndex.cu
    prims/reduce_cols_by_key.cu
    prims/reduce_rows_by_key.cu
    prims/reverse.cu
    prims/rsvd.cu
    prims/score.cu
    prims/seive.cu
    prims/sigmoid.cu
    prims/silhouetteScore.cu
    prims/sqrt.cu
    prims/ternary_op.cu
    prims/trustworthiness.cu
    prims/vMeasure.cu
    prims/weighted_mean.cu
    ../src/common/logger.cpp)  # because prims is header only!

  target_include_directories(prims PRIVATE ${CUML_TEST_INCLUDE_DIRS})

  add_dependencies(prims cutlass)

  target_link_libraries(prims
  	GTest::GTest
  	GTest::Main
  	FAISS::FAISS
    ${PRIMS_TEST_LINK_LIBRARIES})

endif(BUILD_PRIMS_TESTS)
