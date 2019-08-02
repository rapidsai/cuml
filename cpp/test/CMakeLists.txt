# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0079 NEW)

project(cuml_test LANGUAGES CXX CUDA)

# Policy CMP0079 set as NEW as of 3.13 allows googletest be built outside this folder
include_directories(${GTEST_DIR}/googletest/include)

###################################################################################################
# - build ml_test executable ----------------------------------------------------------------

if(BUILD_CUML_TESTS)

    set(ML_TEST_LINK_LIBRARIES
        ${GTEST_LIBNAME}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_nvgraph_LIBRARY}
        gpufaisslib
        faisslib
        ${CUML_CPP_TARGET}
        ${CUML_C_TARGET}
        pthread
        ${ZLIB_LIBRARIES}
    )

    # (please keep the filenames in alphabetical order)
    add_executable(ml
      sg/cd_test.cu
      sg/dbscan_test.cu
      sg/fil_test.cu
      sg/handle_test.cu
      sg/holtwinters_test.cu
      sg/kmeans_test.cu
      sg/knn_test.cu
      sg/lkf_test.cu
      sg/ols.cu
      sg/pca_test.cu
      sg/quasi_newton.cu
      sg/rf_test.cu
      sg/ridge.cu
      sg/rproj_test.cu
      sg/sgd.cu
      sg/spectral_test.cu
      sg/tsne_test.cu
      sg/tsvd_test.cu
      sg/umap_test.cu
      )

    add_dependencies(ml ${ClangFormat_TARGET})

    target_link_libraries(ml
      ${GTEST_LIBNAME}
      ${ML_TEST_LINK_LIBRARIES})

endif(BUILD_CUML_TESTS)

###################################################################################################
# - build test_ml_mg executable ----------------------------------------------------------------

if(BUILD_CUML_MG_TESTS)

    set(ML_MG_TEST_LINK_LIBRARIES
        ${GTEST_LIBNAME}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_nvgraph_LIBRARY}
        gpufaisslib
        faisslib
        ${CUML_CPP_TARGET}
        pthread
        ${ZLIB_LIBRARIES}
    )

    # (please keep the filenames in alphabetical order)
    add_executable(ml_mg
      mg/knn_test_mg.cu
      mg/test_ml_mg_utils.cu
      )

    add_dependencies(ml_mg ${ClangFormat_TARGET})

    target_link_libraries(ml_mg
      ${GTEST_LIBNAME}
      ${ML_MG_TEST_LINK_LIBRARIES})

endif(BUILD_CUML_MG_TESTS)

###################################################################################################
# - build prims_test executable ----------------------------------------------------------------

if(BUILD_PRIMS_TESTS)

    set(PRIMS_LINK_LIBRARIES
        ${CUDA_cublas_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        pthread
        gpufaisslib
        faisslib
        ${ZLIB_LIBRARIES}
    )

    # (please keep the filenames in alphabetical order)
    add_executable(prims
      prims/add.cu
      prims/add_sub_dev_scalar.cu
      prims/adjustedRandIndex.cu
      prims/binary_op.cu
      prims/ternary_op.cu
      prims/coalesced_reduction.cu
      prims/cuda_utils.cu
      prims/columnSort.cu
      prims/completenessScore.cu
      prims/contingencyMatrix.cu
      prims/coo.cu
      prims/cov.cu
      prims/csr.cu
      prims/decoupled_lookback.cu
      prims/dist_adj.cu
      prims/dist_cos.cu
      prims/dist_eps.cu
      prims/dist_euc_exp.cu
      prims/dist_euc_unexp.cu
      prims/dist_l1.cu
      prims/divide.cu
      prims/eig.cu
      prims/eltwise.cu
      prims/eltwise2d.cu
      prims/entropy.cu
      prims/gather.cu
      prims/gemm.cu
      prims/gram.cu
      prims/grid_sync.cu
      prims/hinge.cu
      prims/homogeneityScore.cu
      prims/host_buffer.cu
      prims/jones_transform.cu
      prims/klDivergence.cu
      prims/knn.cu
      prims/kselection.cu
      prims/label.cu
      prims/linearReg.cu
      prims/log.cu
      prims/logisticReg.cu
      prims/make_blobs.cu
      prims/map_then_reduce.cu
      prims/math.cu
      prims/matrix.cu
      prims/matrix_vector_op.cu
      prims/mean.cu
      prims/mean_center.cu
      prims/minmax.cu
      prims/mvg.cu
      prims/multiply.cu
      prims/mutualInfoScore.cu
      prims/norm.cu
      prims/penalty.cu
      prims/permute.cu
      prims/power.cu
      prims/randIndex.cu
      prims/reduce.cu
      prims/reduce_cols_by_key.cu
      prims/reduce_rows_by_key.cu
      prims/reverse.cu
      prims/rng.cu
      prims/rng_int.cu
      prims/rsvd.cu
      prims/sample_without_replacement.cu
      prims/scatter.cu
      prims/score.cu
      prims/sigmoid.cu
      prims/silhouetteScore.cu
      prims/sqrt.cu
      prims/stddev.cu
      prims/strided_reduction.cu
      prims/subtract.cu
      prims/sum.cu
      prims/svd.cu
      prims/transpose.cu
      prims/trustworthiness.cu
      prims/unary_op.cu
      prims/vMeasure.cu
      prims/weighted_mean.cu
      )

    add_dependencies(prims ${ClangFormat_TARGET})

    target_link_libraries(prims
      ${GTEST_LIBNAME}
      ${PRIMS_LINK_LIBRARIES})

endif(BUILD_PRIMS_TESTS)
