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

#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <random/rng.h>
#include <matrix/gather.h>
#include "test_utils.h"

namespace MLCommon {
  namespace Matrix {

    template <typename MatrixIteratorT,
	      typename MapIteratorT>
    __global__
    void naiveGatherKernel(MatrixIteratorT in,
			   int D,
			   int N,
			   MapIteratorT map,
			   int map_length,
			   MatrixIteratorT out){
      int outRow = blockIdx.x * blockDim.x + threadIdx.x;
      if(outRow < map_length){
	typename std::iterator_traits<MapIteratorT>::value_type map_val = map[outRow];
	int inRowStart =  map_val * D;
	int outRowStart =  outRow * D;
	for (int i = 0; i < D; ++i) {
	  out[outRowStart + i] = in[inRowStart + i];
	}
      }
    }

    template <typename MatrixIteratorT,
	      typename MapIteratorT>
    void
    naiveGather(MatrixIteratorT in,
		int D,
		int N,
		MapIteratorT map,
		int map_length,
		MatrixIteratorT out,
		cudaStream_t stream){
      static const int TPB = 64;
      int nblocks = ceildiv(N, TPB);

      naiveGatherKernel<<<nblocks, TPB, 0, stream>>>(in, D, N, map, map_length, out);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    template <typename MatrixIteratorT,
	      typename MapIteratorT>
    void
    gatherLaunch(MatrixIteratorT in,
		 int D,
		 int N,
		 MapIteratorT map,
		 int map_length,
		 MatrixIteratorT out,
		 cudaStream_t stream){
      typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
      Matrix::gather(in, D, N,
		     map,
		     map_length,
		     out,
		     stream);
    }
    
    struct GatherInputs {
      uint32_t nrows;
      uint32_t ncols;
      uint32_t map_length;
      unsigned long long int seed;
    };

    template <typename MatrixT,
	      typename MapT>
    class GatherTest:public ::testing::TestWithParam<GatherInputs> {
    protected: 

      void SetUp() override {
	params = ::testing::TestWithParam<GatherInputs>::GetParam();
	Random::Rng r(params.seed);
	Random::Rng r_int(params.seed);
	CUDA_CHECK(cudaStreamCreate(&stream));
	
	uint32_t nrows = params.nrows;
	uint32_t ncols = params.ncols;
	uint32_t map_length = params.map_length;
	uint32_t len = nrows * ncols;
	
	// input matrix setup
	allocate(in, nrows * ncols);
	r.uniform(in, len, MatrixT(-1.0), MatrixT(1.0));

	// map setup
	allocate(map, map_length);
	r_int.uniformInt(map, map_length, (MapT)0, nrows);

	// expected and actual output matrix setup
	allocate(out_exp, map_length * ncols);
	allocate(out_act, map_length * ncols);

	naiveGather(in, ncols, nrows, map, map_length, out_exp, stream);
	gatherLaunch(in, ncols, nrows, map, map_length, out_act, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      void TearDown() override {
	CUDA_CHECK(cudaFree(in));
	CUDA_CHECK(cudaFree(map));
	CUDA_CHECK(cudaFree(out_exp));
	CUDA_CHECK(cudaFree(out_act));
	CUDA_CHECK(cudaStreamDestroy(stream));
      }
    protected: 
      cudaStream_t stream;
      GatherInputs params;
      MatrixT *in, *out_exp, *out_act;
      MapT *map;
    };

    
    const std::vector<GatherInputs> inputs = {
      {1024, 32, 64, 1234ULL},
      {1024, 32, 128, 1234ULL},
      {1024, 32, 256, 1234ULL},
      {1024, 32, 512, 1234ULL},
      {1024, 32, 1024, 1234ULL},
      {1024, 64, 64, 1234ULL},
      {1024, 64, 128, 1234ULL},
      {1024, 64, 256, 1234ULL},
      {1024, 64, 512, 1234ULL},
      {1024, 64, 1024, 1234ULL},
      {1024, 128, 64, 1234ULL},
      {1024, 128, 128, 1234ULL},
      {1024, 128, 256, 1234ULL},
      {1024, 128, 512, 1234ULL},
      {1024, 128, 1024, 1234ULL}
    };

    typedef GatherTest<float, uint32_t> GatherTestF;
    TEST_P(GatherTestF, Result) {
      ASSERT_TRUE(devArrMatch(out_exp, out_act, params.map_length * params.ncols,
			      Compare<float>()));
    }

    typedef GatherTest<double, uint32_t> GatherTestD;
    TEST_P(GatherTestD, Result) {
      ASSERT_TRUE(devArrMatch(out_exp, out_act, params.map_length * params.ncols,
			      Compare<double>()));
    }
    

    INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestF, ::testing::ValuesIn(inputs));
    INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestD, ::testing::ValuesIn(inputs));

  } // end namespace Matrix
} // end namespace MLCommon

