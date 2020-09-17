/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <raft/linalg/cublas_wrappers.h>
#include <test_utils.h>
#include <cuda_utils.cuh>
#include <cuml/common/logger.hpp>
#include <cuml/decomposition/pca_mg.hpp>
#include <matrix/matrix.cuh>
#include <opg/linalg/gemm.hpp>
#include <opg/matrix/matrix_utils.hpp>
#include "test_opg_utils.h"

namespace MLCommon {
namespace Test {
namespace opg {

struct PCAOpgParams {
  int M;
  int N;
  int N_components;
  ML::mg_solver algorithm;
  std::vector<int> partSizes;
  std::vector<int> ranksOwners;
  Matrix::Layout layout;
  unsigned long long int seed;
};

template <typename T>
class PCAOpgTest : public testing::TestWithParam<PCAOpgParams> {
 public:
  void SetUp() {
    params = GetParam();
    handle = new raft::handle_t();
    ML::initialize_mpi_comms(*handle, MPI_COMM_WORLD);

    // Prepare resource
    const raft::handle_t& h = handle;
    const cumlCommunicator& comm = h.getCommunicator();
    stream = h.get_stream();
    const auto allocator = h.get_device_allocator();
    cublasHandle_t cublasHandle = h.get_cublas_handle();

    myRank = comm.getRank();
    totalRanks = comm.getSize();
    Random::Rng r(params.seed + myRank);

    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

    if (myRank == 0) {
      std::cout << "Testing PCA of " << params.M << " x " << params.N
                << " matrix" << std::endl;
    }

    // Prepare X matrix
    std::vector<Matrix::RankSizePair*> totalPartsToRanks;
    for (int i = 0; i < params.partSizes.size(); i++) {
      Matrix::RankSizePair* rspt = new Matrix::RankSizePair(
        params.ranksOwners[i] % totalRanks, params.partSizes[i]);
      totalPartsToRanks.push_back(rspt);
    }
    Matrix::PartDescriptor desc(params.M, params.N, totalPartsToRanks,
                                comm.getRank(), params.layout);
    std::vector<Matrix::Data<T>*> inParts;
    Matrix::opg::allocate(h, inParts, desc, myRank, stream);
    Matrix::opg::randomize(h, r, inParts, desc, myRank, stream, T(10.0),
                           T(20.0));
    h.waitOnUserStream();

    prmsPCA.n_rows = params.M;
    prmsPCA.n_cols = params.N;
    prmsPCA.n_components = params.N_components;
    prmsPCA.whiten = false;
    prmsPCA.n_iterations = 100;
    prmsPCA.tol = 0.01;
    prmsPCA.algorithm = params.algorithm;

    device_buffer<T> components(allocator, stream,
                                prmsPCA.n_components * prmsPCA.n_cols);

    device_buffer<T> explained_var(allocator, stream, prmsPCA.n_components);

    device_buffer<T> explained_var_ratio(allocator, stream,
                                         prmsPCA.n_components);

    device_buffer<T> singular_vals(allocator, stream, prmsPCA.n_components);

    device_buffer<T> mu(allocator, stream, prmsPCA.n_cols);

    device_buffer<T> noise_vars(allocator, stream, prmsPCA.n_components);

    ML::PCA::opg::fit(*handle, inParts, desc, components.data(),
                      explained_var.data(), explained_var_ratio.data(),
                      singular_vals.data(), mu.data(), noise_vars.data(),
                      prmsPCA, false);

    CUML_LOG_DEBUG(MLCommon::arr2Str(singular_vals.data(), params.N_components,
                                     "Singular Vals", stream)
                     .c_str());

    CUML_LOG_DEBUG(MLCommon::arr2Str(explained_var.data(), params.N_components,
                                     "Explained Variance", stream)
                     .c_str());

    CUML_LOG_DEBUG(MLCommon::arr2Str(explained_var_ratio.data(),
                                     params.N_components,
                                     "Explained Variance Ratio", stream)
                     .c_str());

    CUML_LOG_DEBUG(MLCommon::arr2Str(components.data(),
                                     params.N_components * params.N,
                                     "Components", stream)
                     .c_str());

    Matrix::opg::deallocate(h, inParts, desc, myRank, stream);
  }

  void TearDown() { delete handle; }

 protected:
  PCAOpgParams params;
  raft::handle_t* handle;
  cudaStream_t stream;
  int myRank;
  int totalRanks;
  ML::paramsPCAMG prmsPCA;
};

const std::vector<PCAOpgParams> inputs = {{20,
                                           4,
                                           2,
                                           ML::mg_solver::COV_EIG_JACOBI,
                                           {11, 9},
                                           {1, 0},
                                           Matrix::LayoutColMajor,
                                           223548ULL},
                                          {20,
                                           4,
                                           2,
                                           ML::mg_solver::COV_EIG_DQ,
                                           {11, 9},
                                           {1, 0},
                                           Matrix::LayoutColMajor,
                                           223548ULL},
                                          {20,
                                           4,
                                           2,
                                           ML::mg_solver::QR,
                                           {11, 9},
                                           {1, 0},
                                           Matrix::LayoutColMajor,
                                           223548ULL}};

typedef PCAOpgTest<float> PCAOpgTestF;

TEST_P(PCAOpgTestF, Result) {
  if (myRank == 0) {
    // We should be inverse transforming and checking against the original
    // data here. Github reference: https://github.com/rapidsai/cuml/issues/2474

    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestF, ::testing::ValuesIn(inputs));

typedef PCAOpgTest<double> PCAOpgTestD;

TEST_P(PCAOpgTestD, Result) {
  if (myRank == 0) {
    // We should be inverse transforming and checking against the original
    // data here. Github reference: https://github.com/rapidsai/cuml/issues/2474

    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon