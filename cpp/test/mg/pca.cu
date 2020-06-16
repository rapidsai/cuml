/*
* Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <cuda_utils.cuh>
#include "cuml/decomposition/pca_mg.hpp"
#include "cuml/test/prims/test_utils.h"
#include "cumlprims/opg/linalg/gemm.hpp"
#include "cumlprims/opg/matrix/matrix_utils.hpp"
#include "linalg/cublas_wrappers.h"
#include "matrix/matrix.cuh"
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
    handle = new ML::cumlHandle();
    ML::initialize_mpi_comms(*handle, MPI_COMM_WORLD);

    // Prepare resource
    const ML::cumlHandle_impl& h = handle->getImpl();
    const cumlCommunicator& comm = h.getCommunicator();
    stream = h.getStream();
    const std::shared_ptr<deviceAllocator> allocator = h.getDeviceAllocator();
    cublasHandle_t cublasHandle = h.getCublasHandle();

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

    components = (T*)allocator->allocate(
      prmsPCA.n_components * prmsPCA.n_cols * sizeof(T), stream);

    explained_var =
      (T*)allocator->allocate(prmsPCA.n_components * sizeof(T), stream);

    explained_var_ratio =
      (T*)allocator->allocate(prmsPCA.n_components * sizeof(T), stream);

    singular_vals =
      (T*)allocator->allocate(prmsPCA.n_components * sizeof(T), stream);

    mu = (T*)allocator->allocate(prmsPCA.n_cols * sizeof(T), stream);

    noise_vars =
      (T*)allocator->allocate(prmsPCA.n_components * sizeof(T), stream);

    ML::PCA::opg::fit(*handle, inParts, desc, components, explained_var,
                      explained_var_ratio, singular_vals, mu, noise_vars,
                      prmsPCA, false);

    Matrix::opg::deallocate(h, inParts, desc, myRank, stream);
  }

  void TearDown() {
    const std::shared_ptr<deviceAllocator> allocator =
      handle->getDeviceAllocator();
    cudaStream_t stream = handle->getStream();

    allocator->deallocate(
      components, prmsPCA.n_components * prmsPCA.n_cols * sizeof(T), stream);

    allocator->deallocate(explained_var, prmsPCA.n_components * sizeof(T),
                          stream);

    allocator->deallocate(explained_var_ratio, prmsPCA.n_components * sizeof(T),
                          stream);

    allocator->deallocate(singular_vals, prmsPCA.n_components * sizeof(T),
                          stream);

    allocator->deallocate(mu, prmsPCA.n_cols * sizeof(T), stream);

    allocator->deallocate(noise_vars, prmsPCA.n_components * sizeof(T), stream);

    delete handle;
  }

 protected:
  PCAOpgParams params;
  ML::cumlHandle* handle;
  cudaStream_t stream;
  int myRank;
  int totalRanks;
  ML::paramsPCAMG prmsPCA;
  T *components, *explained_var, *explained_var_ratio, *singular_vals, *mu,
    *noise_vars;
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
    MLCommon::myPrintDevVector("Singular Values: ", singular_vals,
                               params.N_components);
    MLCommon::myPrintDevVector("Explained Variance: ", explained_var,
                               params.N_components);
    MLCommon::myPrintDevVector(
      "Explained Variance Ratio: ", explained_var_ratio, params.N_components);
    std::cout << "Components: ";
    MLCommon::Matrix::print(components, params.N_components, params.N);
    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestF, ::testing::ValuesIn(inputs));

typedef PCAOpgTest<double> PCAOpgTestD;

TEST_P(PCAOpgTestD, Result) {
  if (myRank == 0) {
    MLCommon::myPrintDevVector("Singular Values: ", singular_vals,
                               params.N_components);
    MLCommon::myPrintDevVector("Explained Variance: ", explained_var,
                               params.N_components);
    MLCommon::myPrintDevVector(
      "Explained Variance Ratio: ", explained_var_ratio, params.N_components);
    std::cout << "Components: ";
    MLCommon::Matrix::print(components, params.N_components, params.N);
    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon