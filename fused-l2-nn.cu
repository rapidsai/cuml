/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <distance/fused_l2_nn.h>
#include <linalg/norm.h>
#include <random/rng.h>
#include <thrust/reduce.h>
#include <thrust/equal.h>

struct Inputs {
  int m, n, k;
  unsigned long long int seed;
};

namespace MLCommon {
namespace Distance {


template <typename DataT, bool sqrt = false> void test(std::vector<Inputs> inputs) {
  for (auto &params : inputs) {
    DataT *x, *y, *xn, *yn;
    char *workspace;
    DataT *min, *min_ref;
    cudaStream_t stream;

    Random::Rng r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;

    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(x, m * k);
    allocate(y, n * k);
    allocate(xn, m);
    allocate(yn, n);
    allocate(workspace, sizeof(int) * m);
    allocate(min, m);
    allocate(min_ref, m);
    r.uniform(x, m * k, DataT(-1.0), DataT(1.0), stream);
    r.uniform(y, n * k, DataT(-1.0), DataT(1.0), stream);

    

    LinAlg::rowNorm(xn, x, k, m, LinAlg::L2Norm, true, stream);
    LinAlg::rowNorm(yn, y, k, n, LinAlg::L2Norm, true, stream);
    
    fusedL2NN<DataT, DataT, int, sqrt, MinReduceOp<int, DataT>>(
        min_ref, x, y, xn, yn, m, n, k, (void *)workspace, stream);

    DataT avg_inertia = 0, min_inertia = std::numeric_limits<DataT>::max(),
        max_inertia = std::numeric_limits<DataT>::min();
    
    auto nruns = 1000;
    auto nequals = 0;
    for (auto i = 0; i < nruns; ++i) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      fusedL2NN<DataT, DataT, int, sqrt, MinReduceOp<int, DataT>>(
          min, x, y, xn, yn, m, n, k, (void *)workspace, stream);

      DataT inertia = thrust::reduce(thrust::cuda::par.on(stream), min, min + m);
      bool result = thrust::equal(thrust::cuda::par.on(stream), min, min + m, min_ref);
      if(result) ++nequals;
      
      avg_inertia += inertia;
      if (inertia < min_inertia)
        min_inertia = inertia;
      if (inertia > max_inertia)
        max_inertia = inertia;
    }
    std::cout << nequals << " of " << nruns << " runs matches with the reference\n";
    std::cout << " min - " << min_inertia << ", ";
    std::cout << " max - " << max_inertia << ", ";
    std::cout << " avg - " << avg_inertia / nruns << "\n";
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(xn));
    CUDA_CHECK(cudaFree(yn));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(min));
  }
}

} // end namespace Distance
} // end namespace MLCommon

const std::vector<Inputs> inputs = {
    {1805, 134, 2, 1234ULL},
};

int main() {
  MLCommon::Distance::test<float>(inputs);
  MLCommon::Distance::test<double>(inputs);
  return 0;
}
