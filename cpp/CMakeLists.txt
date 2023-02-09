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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUML)

project(CUML VERSION 23.02.00 LANGUAGES CXX CUDA)

# Write the version header
rapids_cmake_write_version_file(include/cuml/version_config.hpp)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuML shared libraries" ON)
option(BUILD_CUML_C_LIBRARY "Build libcuml_c shared library. Contains the cuML C API" ON)
option(BUILD_CUML_CPP_LIBRARY "Build libcuml shared library" ON)
option(BUILD_CUML_TESTS "Build cuML algorithm tests" ON)
option(BUILD_CUML_MG_TESTS "Build cuML multigpu algorithm tests" OFF)
option(BUILD_PRIMS_TESTS "Build ml-prim tests" ON)
option(BUILD_CUML_EXAMPLES "Build C++ API usage examples" ON)
option(BUILD_CUML_BENCH "Build cuML C++ benchmark tests" ON)
option(BUILD_CUML_MPI_COMMS "Build the MPI+NCCL Communicator (used for testing)" OFF)
option(CUDA_ENABLE_KERNEL_INFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINE_INFO "Enable lineinfo in nvcc" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecation warnings " OFF)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(ENABLE_CUMLPRIMS_MG "Enable algorithms that use libcumlprims_mg" ON)
option(NVTX "Enable nvtx markers" OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(USE_CCACHE "Cache build artifacts with ccache" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUML_USE_RAFT_STATIC "Build and statically link the RAFT libraries" OFF)
option(CUML_USE_FAISS_STATIC "Build and statically link the FAISS library for nearest neighbors search on GPU" OFF)
option(CUML_USE_TREELITE_STATIC "Build and statically link the treelite library" OFF)
option(CUML_EXPORT_TREELITE_LINKAGE "Whether to publicly or privately link treelite to libcuml++" OFF)
option(CUML_USE_CUMLPRIMS_MG_STATIC "Build and statically link the cumlprims_mg library" OFF)
# The options below allow incorporating libcuml into another build process
# without installing all its components. This is useful if total file size is
# at a premium and we do not expect other consumers to use any APIs of the
# dependency except those that are directly linked to by the dependent library.
option(CUML_EXCLUDE_RAFT_FROM_ALL "Exclude RAFT targets from cuML's 'all' target" OFF)
option(CUML_EXCLUDE_TREELITE_FROM_ALL "Exclude Treelite targets from cuML's 'all' target" OFF)
option(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL "Exclude cumlprims_mg targets from cuML's 'all' target" OFF)
option(CUML_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" ON)

message(VERBOSE "CUML_CPP: Building libcuml_c shared library. Contains the cuML C API: ${BUILD_CUML_C_LIBRARY}")
message(VERBOSE "CUML_CPP: Building libcuml shared library: ${BUILD_CUML_CPP_LIBRARY}")
message(VERBOSE "CUML_CPP: Building cuML algorithm tests: ${BUILD_CUML_TESTS}")
message(VERBOSE "CUML_CPP: Building cuML multigpu algorithm tests: ${BUILD_CUML_MG_TESTS}")
message(VERBOSE "CUML_CPP: Building ml-prims tests: ${BUILD_PRIMS_TESTS}")
message(VERBOSE "CUML_CPP: Building C++ API usage examples: ${BUILD_CUML_EXAMPLES}")
message(VERBOSE "CUML_CPP: Building cuML C++ benchmark tests: ${BUILD_CUML_BENCH}")
message(VERBOSE "CUML_CPP: Building the MPI+NCCL Communicator (used for testing): ${BUILD_CUML_MPI_COMMS}")
message(VERBOSE "CUML_CPP: Enabling detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "CUML_CPP: Disabling OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "CUML_CPP: Enabling algorithms that use libcumlprims_mg: ${ENABLE_CUMLPRIMS_MG}")
message(VERBOSE "CUML_CPP: Enabling kernel resource usage info: ${KERNEL_INFO}")
message(VERBOSE "CUML_CPP: Enabling kernelinfo in nvcc: ${CUDA_ENABLE_KERNEL_INFO}")
message(VERBOSE "CUML_CPP: Enabling lineinfo in nvcc: ${CUDA_ENABLE_LINE_INFO}")
message(VERBOSE "CUML_CPP: Enabling nvtx markers: ${NVTX}")
message(VERBOSE "CUML_CPP: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")
message(VERBOSE "CUML_CPP: Cache build artifacts with ccache: ${USE_CCACHE}")
message(VERBOSE "CUML_CPP: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "CUML_CPP: Build and statically link RAFT libraries: ${CUML_USE_RAFT_STATIC}")
message(VERBOSE "CUML_CPP: Build and statically link FAISS library: ${CUML_USE_FAISS_STATIC}")
message(VERBOSE "CUML_CPP: Build and statically link Treelite library: ${CUML_USE_TREELITE_STATIC}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Experimental: Choose which algorithms are built into libcuml++.so. Can specify individual algorithms or groups in a semicolon-separated list.")
message(VERBOSE "CUML_CPP: Building libcuml++ with algoriths: '${CUML_ALGORITHMS}'.")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "CUML_CPP: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if(BUILD_CUML_TESTS OR BUILD_PRIMS_TESTS)
  # Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
  # have different values for the `Threads::Threads` target. Setting this flag ensures
  # `Threads::Threads` is the same value in first run and subsequent runs.
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

##############################################################################
# - Target names -------------------------------------------------------------

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_BENCH_TARGET "sg_benchmark")
if(${BUILD_CUML_C_LIBRARY})
  set(CUML_C_TARGET "cuml")
endif()
set(CUML_C_TEST_TARGET "${CUML_C_TARGET}_test")
set(CUML_MG_TEST_TARGET "ml_mg")
set(PRIMS_BENCH_TARGET "prims_benchmark")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "CUML_CPP: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static_nocallback")
endif()

if (NOT DISABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "CUML_CPP: OpenMP found in ${OPENMP_INCLUDE_DIRS}")
  endif()
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cuml-exports
    INSTALL_EXPORT_SET cuml-exports
    )
include(cmake/modules/ConfigureCUDA.cmake)


##############################################################################
# - Set options based on user defined one  -----------------------------------
set(CUML_USE_RAFT_NN OFF)
set(CUML_USE_RAFT_DIST OFF)
set(LINK_TREELITE OFF)
set(LINK_CUFFT OFF)
include(cmake/modules/ConfigureAlgorithms.cmake)

# Enabling libcuml enables building libcuml++
if(BUILD_CUML_C_LIBRARY)
  set(BUILD_CUML_CPP_LIBRARY ON)
endif()

# Disabling libcuml++ disables building algorithm tests and examples
if(NOT BUILD_CUML_CPP_LIBRARY)
  set(BUILD_CUML_C_LIBRARY OFF)
  set(BUILD_CUML_TESTS OFF)
  set(BUILD_CUML_MG_TESTS OFF)
  set(BUILD_CUML_EXAMPLES OFF)
endif()

# SingleGPU build disables cumlprims_mg and comms components
if(SINGLEGPU)
  message(STATUS "CUML_CPP: Detected SINGLEGPU build option")
  message(STATUS "CUML_CPP: Disabling Multi-GPU components and comms libraries")
  set(BUILD_CUML_MG_TESTS OFF)
  set(BUILD_CUML_MPI_COMMS OFF)
  set(ENABLE_CUMLPRIMS_MG OFF)
  set(WITH_UCX OFF)
endif()

if(BUILD_CUML_MG_TESTS AND NOT SINGLEGPU)
  message(STATUS "CUML_CPP: Detected BUILD_CUML_MG_TESTS set to ON. Enabling BUILD_CUML_MPI_COMMS")
  set(BUILD_CUML_MPI_COMMS ON)
endif()

if(USE_CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

##############################################################################
# - Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()
rapids_cmake_install_lib_dir(lib_dir)

if(BUILD_CUML_TESTS OR BUILD_PRIMS_TESTS)
  find_package(Threads)
endif()

include(cmake/thirdparty/get_raft.cmake)

if(LINK_TREELITE)
  include(cmake/thirdparty/get_treelite.cmake)
endif()

if(all_algo OR treeshap_algo)
  include(cmake/thirdparty/get_gputreeshap.cmake)
  # Workaround until https://github.com/rapidsai/rapids-cmake/issues/176 is resolved
  if(NOT BUILD_SHARED_LIBS)
    rapids_export_package(BUILD GPUTreeShap cuml-exports)
    rapids_export_package(INSTALL GPUTreeShap cuml-exports)
  endif()
endif()

if(ENABLE_CUMLPRIMS_MG)
  include(cmake/thirdparty/get_cumlprims_mg.cmake)
endif()

if(BUILD_CUML_TESTS OR BUILD_PRIMS_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

if(BUILD_CUML_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()
endif()

##############################################################################
# - build libcuml++ shared library -------------------------------------------

if(BUILD_CUML_C_LIBRARY OR BUILD_CUML_CPP_LIBRARY)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])
endif()

# Copy the interface include directories from INCLUDED_TARGET to TARGET.
# This is necessary when INCLUDED_TARGET was compiled statically but includes
# public APIs that may still require consumers to have the same interface
# headers available.
function(copy_interface_excludes)
  set(_options "")
  set(_one_value TARGET INCLUDED_TARGET)
  set(_multi_value "")
  cmake_parse_arguments(_CUML_INCLUDES "${_options}" "${_one_value}"
                        "${_multi_value}" ${ARGN})
  get_target_property(_includes ${_CUML_INCLUDES_INCLUDED_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(${_CUML_INCLUDES_TARGET} PUBLIC ${_includes})
endfunction()

if(BUILD_CUML_CPP_LIBRARY)

  # single GPU components
  # common components
  add_library(${CUML_CPP_TARGET}
              src/common/logger.cpp)

  if(all_algo OR arima_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/arima/batched_arima.cu
        src/arima/batched_kalman.cu)
  endif()

  if(all_algo OR datasets_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/datasets/make_arima.cu
        src/datasets/make_blobs.cu
        src/datasets/make_regression.cu)
  endif()

  if(all_algo OR dbscan_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/dbscan/dbscan.cu)
  endif()

  if(all_algo OR decisiontree_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/decisiontree/batched-levelalgo/kernels/entropy-double.cu
        src/decisiontree/batched-levelalgo/kernels/entropy-float.cu
        src/decisiontree/batched-levelalgo/kernels/gamma-double.cu
        src/decisiontree/batched-levelalgo/kernels/gamma-float.cu
        src/decisiontree/batched-levelalgo/kernels/gini-double.cu
        src/decisiontree/batched-levelalgo/kernels/gini-float.cu
        src/decisiontree/batched-levelalgo/kernels/inverse_gaussian-double.cu
        src/decisiontree/batched-levelalgo/kernels/inverse_gaussian-float.cu
        src/decisiontree/batched-levelalgo/kernels/mse-double.cu
        src/decisiontree/batched-levelalgo/kernels/mse-float.cu
        src/decisiontree/batched-levelalgo/kernels/poisson-double.cu
        src/decisiontree/batched-levelalgo/kernels/poisson-float.cu
        src/decisiontree/batched-levelalgo/kernels/quantiles.cu
        src/decisiontree/decisiontree.cu)
  endif()

  if(all_algo OR explainer_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/explainer/kernel_shap.cu
        src/explainer/permutation_shap.cu)
  endif()

  if(all_algo OR treeshap_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/explainer/tree_shap.cu)
  endif()

  # FIL components
  if(all_algo OR fil_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/fil/fil.cu
        src/fil/infer.cu
        src/fil/treelite_import.cu)
  endif()

  # todo: organize linear models better
  if(all_algo OR linearregression_algo OR ridge_algo OR lasso_algo OR logisticregression_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/glm/glm.cu)
  endif()

  if(all_algo OR genetic_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/genetic/genetic.cu
        src/genetic/node.cu)
  endif()

  if(all_algo OR hdbscan_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/genetic/program.cu
        src/genetic/node.cu
        src/hdbscan/hdbscan.cu
        src/hdbscan/condensed_hierarchy.cu
        src/hdbscan/prediction_data.cu)
  endif()

  if(all_algo OR holtwinters_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/holtwinters/holtwinters.cu)
  endif()

  if(all_algo OR kmeans_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/kmeans/kmeans.cu)
  endif()

  if(all_algo OR knn_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/knn/knn.cu
        src/knn/knn_sparse.cu)
  endif()

  if(all_algo OR hierarchicalclustering_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/hierarchy/linkage.cu)
  endif()

  if(all_algo OR metrics_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/metrics/accuracy_score.cu
        src/metrics/adjusted_rand_index.cu
        src/metrics/completeness_score.cu
        src/metrics/entropy.cu
        src/metrics/homogeneity_score.cu
        src/metrics/kl_divergence.cu
        src/metrics/mutual_info_score.cu
        src/metrics/pairwise_distance.cu
        src/metrics/pairwise_distance_canberra.cu
        src/metrics/pairwise_distance_chebyshev.cu
        src/metrics/pairwise_distance_correlation.cu
        src/metrics/pairwise_distance_cosine.cu
        src/metrics/pairwise_distance_euclidean.cu
        src/metrics/pairwise_distance_hamming.cu
        src/metrics/pairwise_distance_hellinger.cu
        src/metrics/pairwise_distance_jensen_shannon.cu
        src/metrics/pairwise_distance_kl_divergence.cu
        src/metrics/pairwise_distance_l1.cu
        src/metrics/pairwise_distance_minkowski.cu
        src/metrics/pairwise_distance_russell_rao.cu
        src/metrics/r2_score.cu
        src/metrics/rand_index.cu
        src/metrics/silhouette_score.cu
        src/metrics/silhouette_score_batched_double.cu
        src/metrics/silhouette_score_batched_float.cu
        src/metrics/trustworthiness.cu
        src/metrics/v_measure.cu)
  endif()

  if(all_algo OR pca_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/pca/pca.cu)
  endif()

  if(all_algo OR randomforest_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/randomforest/randomforest.cu)
  endif()

  if(all_algo OR randomprojection_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/random_projection/rproj.cu)
  endif()

  # todo: separate solvers better
  if(all_algo OR solvers_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/solver/lars.cu
        src/solver/solver.cu)
  endif()

  if(all_algo OR spectralclustering_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/spectral/spectral.cu)
  endif()

  if(all_algo OR svm_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/svm/svc.cu
        src/svm/svr.cu
        src/svm/linear.cu
        src/svm/ws_util.cu)
  endif()

  if(all_algo OR autoarima_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/tsa/auto_arima.cu
        src/tsa/stationarity.cu)
  endif()

  if(all_algo OR tsne_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/tsne/tsne.cu)
  endif()

  if(all_algo OR tsvd_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/tsvd/tsvd.cu)
  endif()

  if(all_algo OR umap_algo)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/umap/umap.cu)
  endif()

  # multi GPU components
  # todo: separate mnmg that require cumlprims from those that don't
  if(NOT SINGLEGPU)
    target_sources(${CUML_CPP_TARGET}
      PRIVATE
        src/glm/ols_mg.cu
        src/glm/preprocess_mg.cu
        src/glm/ridge_mg.cu
        src/kmeans/kmeans_mg.cu
        src/knn/knn_mg.cu
        src/knn/knn_classify_mg.cu
        src/knn/knn_regress_mg.cu
        src/pca/pca_mg.cu
        src/pca/sign_flip_mg.cu
        src/solver/cd_mg.cu
        src/tsvd/tsvd_mg.cu
    )
  endif()

  add_library(cuml::${CUML_CPP_TARGET} ALIAS ${CUML_CPP_TARGET})

  set_target_properties(${CUML_CPP_TARGET}
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  target_compile_definitions(${CUML_CPP_TARGET}
    PUBLIC
      DISABLE_CUSPARSE_DEPRECATED
    PRIVATE
      CUML_CPP_API
  )

  target_compile_options(${CUML_CPP_TARGET}
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUML_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${CUML_CUDA_FLAGS}>"
  )

  target_include_directories(${CUML_CPP_TARGET}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/metrics>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src_prims>
      $<$<BOOL:${BUILD_CUML_MPI_COMMS}>:${MPI_CXX_INCLUDE_PATH}>
    INTERFACE
      $<INSTALL_INTERFACE:include>
  )

  set(_cuml_cpp_public_libs)
  set(_cuml_cpp_private_libs)

  if(CUML_USE_RAFT_STATIC AND (TARGET raft::raft))
    copy_interface_excludes(INCLUDED_TARGET raft::raft TARGET ${CUML_CPP_TARGET})

    if(CUML_USE_RAFT_NN AND (TARGET faiss::faiss))
      copy_interface_excludes(INCLUDED_TARGET faiss::faiss TARGET ${CUML_CPP_TARGET})
    endif()

    if(CUML_USE_RAFT_DIST AND (TARGET cuco::cuco))
      list(APPEND _cuml_cpp_private_libs cuco::cuco)
    endif()
  endif()

  if(CUML_USE_TREELITE_STATIC AND (TARGET treelite::treelite_static))
    # By default, TREELITE_LIBS will contain both treelite::treelite_static and
    # treelite::treelite_runtime_static if we are linking statically, but these
    # two targets have duplicate symbols so we can only link to one of them.
    set(TREELITE_LIBS treelite::treelite_static)

    copy_interface_excludes(INCLUDED_TARGET treelite::treelite_static TARGET ${CUML_CPP_TARGET})
  elseif(CUML_EXPORT_TREELITE_LINKAGE)
    list(APPEND _cuml_cpp_public_libs ${TREELITE_LIBS})
  endif()

  if(CUML_USE_CUMLPRIMS_MG_STATIC AND (TARGET cumlprims_mg::cumlprims_mg))
    copy_interface_excludes(INCLUDED_TARGET cumlprims_mg::cumlprims_mg TARGET ${CUML_CPP_TARGET})
  endif()

  # These are always private:
  list(APPEND _cuml_cpp_private_libs
    $<TARGET_NAME_IF_EXISTS:GPUTreeShap::GPUTreeShap>
    $<$<BOOL:${LINK_CUFFT}>:CUDA::cufft${_ctk_static_suffix}>
    ${TREELITE_LIBS}
    ${OpenMP_CXX_LIB_NAMES}
    $<$<OR:$<BOOL:${BUILD_CUML_STD_COMMS}>,$<BOOL:${BUILD_CUML_MPI_COMMS}>>:NCCL::NCCL>
    $<$<BOOL:${BUILD_CUML_MPI_COMMS}>:${MPI_CXX_LIBRARIES}>
  )

  set(_cuml_cpp_libs_var_name "_cuml_cpp_public_libs")
  if(CUDA_STATIC_RUNTIME)
    set(_cuml_cpp_libs_var_name "_cuml_cpp_private_libs")
    # Add CTK include paths because we're going to make our CTK library links private below
    target_include_directories(${CUML_CPP_TARGET} SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  endif()

  # The visibility of these depend on whether we're linking the CTK statically,
  # because cumlprims_mg and cuML inherit their CUDA libs from the raft::raft
  # INTERFACE target.
  list(APPEND ${_cuml_cpp_libs_var_name}
    raft::raft
    $<$<BOOL:${CUML_USE_RAFT_NN}>:raft::nn>
    $<$<BOOL:${CUML_USE_RAFT_DIST}>:raft::distance>
    $<TARGET_NAME_IF_EXISTS:cumlprims_mg::cumlprims_mg>
  )

  target_link_libraries(${CUML_CPP_TARGET}
    PUBLIC  rmm::rmm
            ${_cuml_cpp_public_libs}
    PRIVATE ${_cuml_cpp_private_libs}
  )

  # If we export the libdmlc symbols, they can lead to weird crashes with other
  # libraries that use libdmlc. This just hides the symbols internally.
  target_link_options(${CUML_CPP_TARGET} PRIVATE "-Wl,--exclude-libs,libdmlc.a")
  # same as above, but for protobuf library
  target_link_options(${CUML_CPP_TARGET} PRIVATE "-Wl,--exclude-libs,libprotobuf.a")
  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(${CUML_CPP_TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

#############################################################################
# - build libcuml C shared library -------------------------------------------

if(BUILD_CUML_C_LIBRARY)
  add_library(${CUML_C_TARGET}
    src/common/cumlHandle.cpp
    src/common/cuml_api.cpp
    src/dbscan/dbscan_api.cpp
    src/glm/glm_api.cpp
    src/holtwinters/holtwinters_api.cpp
    src/knn/knn_api.cpp
    src/svm/svm_api.cpp
  )

  add_library(cuml::${CUML_C_TARGET} ALIAS ${CUML_C_TARGET})

  target_compile_definitions(${CUML_C_TARGET}
    PRIVATE
      CUML_C_API)

  target_include_directories(${CUML_C_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

  target_link_libraries(${CUML_C_TARGET}
    PUBLIC
      ${CUML_CPP_TARGET}
  )

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(${CUML_C_TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

endif()

##############################################################################
# - build test executables ---------------------------------------------------

if(BUILD_CUML_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()

##############################################################################
# - build examples -----------------------------------------------------------

if(BUILD_CUML_EXAMPLES)
  add_subdirectory(examples)
endif()

# ###################################################################################################
# # - install targets -------------------------------------------------------------------------------
include(CPack)

set(CUML_TARGETS ${CUML_CPP_TARGET})

if(BUILD_CUML_C_LIBRARY)
  list(APPEND CUML_TARGETS
         ${CUML_C_TARGET})
endif()

install(TARGETS
          ${CUML_TARGETS}
        DESTINATION
          ${lib_dir}
        EXPORT
          cuml-exports)

install(DIRECTORY include/cuml/
        DESTINATION include/cuml)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuml/version_config.hpp
        DESTINATION include/cuml)

# ################################################################################################
# # - install export -------------------------------------------------------------------------------
set(doc_string
[=[
Provide targets for cuML.

cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives
functions that share compatible APIs with other RAPIDS projects.

]=])

set(code_string )

if (TARGET treelite::treelite)
    string(APPEND code_string
[=[
if (TARGET treelite::treelite AND (NOT TARGET treelite))
    add_library(treelite ALIAS treelite::treelite)
endif()
if (TARGET treelite::treelite_runtime AND (NOT TARGET treelite_runtime))
    add_library(treelite_runtime ALIAS treelite::treelite_runtime)
endif()
]=])
else()
    string(APPEND code_string
[=[
if (TARGET treelite::treelite_static AND (NOT TARGET treelite_static))
    add_library(treelite_static ALIAS treelite::treelite_static)
endif()
if (TARGET treelite::treelite_runtime_static AND (NOT TARGET treelite_runtime_static))
    add_library(treelite_runtime_static ALIAS treelite::treelite_runtime_static)
endif()
]=])

endif()

rapids_export(INSTALL cuml
    EXPORT_SET cuml-exports
    GLOBAL_TARGETS ${CUML_C_TARGET} ${CUML_CPP_TARGET}
    NAMESPACE cuml::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string
    )

################################################################################################
# - build export -------------------------------------------------------------------------------

rapids_export(BUILD cuml
    EXPORT_SET cuml-exports
    GLOBAL_TARGETS ${CUML_C_TARGET} ${CUML_CPP_TARGET}
    NAMESPACE cuml::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string
    )

##############################################################################
# - build benchmark executable -----------------------------------------------

if(BUILD_CUML_BENCH)
  add_subdirectory(bench)
endif()

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
