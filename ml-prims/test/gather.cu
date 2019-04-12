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
void naiveGatherImpl(MatrixIteratorT in,
		     int D,
		     int N,
		     MapIteratorT map,
		     int map_length,
		     MatrixIteratorT out){
  for(int outRow = 0; outRow < map_length; ++outRow){
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
	    MatrixIteratorT out){
  naiveGatherImpl(in, D, N, map, map_length, out);
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
    allocate(d_in, nrows * ncols);
    h_in = (MatrixT *)malloc(sizeof(MatrixT) * nrows * ncols);	
    r.uniform(d_in, len, MatrixT(-1.0), MatrixT(1.0));
    updateHost(h_in, d_in, len, stream);
	
    // map setup
    allocate(d_map, map_length);
    h_map = (MapT *)malloc(sizeof(MapT) * map_length);
    r_int.uniformInt(d_map, map_length, (MapT)0, nrows);
    updateHost(h_map, d_map, map_length, stream);
	
    // expected and actual output matrix setup
    h_out = (MatrixT *)malloc(sizeof(MatrixT) * map_length * ncols);
    allocate(d_out_exp, map_length * ncols);
    allocate(d_out_act, map_length * ncols);

    // launch gather on the host and copy the results to device
    naiveGather(h_in, ncols, nrows, h_map, map_length, h_out);
    updateDevice(d_out_exp, h_out, map_length * ncols);
	
    // launch device version of the kernel
    gatherLaunch(d_in, ncols, nrows, d_map, map_length, d_out_act, stream);


    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  void TearDown() override {
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_map));
    CUDA_CHECK(cudaFree(d_out_act));
    CUDA_CHECK(cudaFree(d_out_exp));

    free(h_in);
    free(h_map);
    free(h_out);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
protected: 
  cudaStream_t stream;
  GatherInputs params;
  MatrixT *d_in, *h_in, *d_out_exp, *d_out_act, *h_out;
  MapT *d_map, *h_map;
};

    
const std::vector<GatherInputs> inputs = {
  {1024, 32, 128, 1234ULL},
  {1024, 32, 256, 1234ULL},
  {1024, 32, 512, 1234ULL},
  {1024, 32, 1024, 1234ULL},
  {1024, 64, 128, 1234ULL},
  {1024, 64, 256, 1234ULL},
  {1024, 64, 512, 1234ULL},
  {1024, 64, 1024, 1234ULL},
  {1024, 128, 128, 1234ULL},
  {1024, 128, 256, 1234ULL},
  {1024, 128, 512, 1234ULL},
  {1024, 128, 1024, 1234ULL}
};

typedef GatherTest<float, uint32_t> GatherTestF;
TEST_P(GatherTestF, Result) {
  ASSERT_TRUE(devArrMatch(d_out_exp, d_out_act, params.map_length * params.ncols,
			  Compare<float>()));
}

typedef GatherTest<double, uint32_t> GatherTestD;
TEST_P(GatherTestD, Result) {
  ASSERT_TRUE(devArrMatch(d_out_exp, d_out_act, params.map_length * params.ncols,
			  Compare<double>()));
}
    

INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestF, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestD, ::testing::ValuesIn(inputs));

} // end namespace Matrix
} // end namespace MLCommon

