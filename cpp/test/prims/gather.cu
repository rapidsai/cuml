/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <gtest/gtest.h>
#include <matrix/gather.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Matrix {

template <typename MatrixIteratorT, typename MapIteratorT>
void naiveGatherImpl(
  MatrixIteratorT in, int D, int N, MapIteratorT map, int map_length, MatrixIteratorT out)
{
  for (int outRow = 0; outRow < map_length; ++outRow) {
    typename std::iterator_traits<MapIteratorT>::value_type map_val = map[outRow];
    int inRowStart                                                  = map_val * D;
    int outRowStart                                                 = outRow * D;
    for (int i = 0; i < D; ++i) {
      out[outRowStart + i] = in[inRowStart + i];
    }
  }
}

template <typename MatrixIteratorT, typename MapIteratorT>
void naiveGather(
  MatrixIteratorT in, int D, int N, MapIteratorT map, int map_length, MatrixIteratorT out)
{
  naiveGatherImpl(in, D, N, map, map_length, out);
}

template <typename MatrixIteratorT, typename MapIteratorT>
void gatherLaunch(MatrixIteratorT in,
                  int D,
                  int N,
                  MapIteratorT map,
                  int map_length,
                  MatrixIteratorT out,
                  cudaStream_t stream)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  Matrix::gather(in, D, N, map, map_length, out, stream);
}

struct GatherInputs {
  uint32_t nrows;
  uint32_t ncols;
  uint32_t map_length;
  unsigned long long int seed;
};

template <typename MatrixT, typename MapT>
class GatherTest : public ::testing::TestWithParam<GatherInputs> {
 protected:
  GatherTest() : d_in(0, stream), d_out_exp(0, stream), d_out_act(0, stream), d_map(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<GatherInputs>::GetParam();
    raft::random::Rng r(params.seed);
    raft::random::Rng r_int(params.seed);
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    uint32_t nrows      = params.nrows;
    uint32_t ncols      = params.ncols;
    uint32_t map_length = params.map_length;
    uint32_t len        = nrows * ncols;

    // input matrix setup
    d_in.resize(nrows * ncols, stream);
    h_in.resize(nrows * ncols);
    r.uniform(d_in.data(), len, MatrixT(-1.0), MatrixT(1.0), stream);
    raft::update_host(h_in.data(), d_in.data(), len, stream);

    // map setup
    d_map.resize(map_length, stream);
    h_map.resize(map_length);
    r_int.uniformInt(d_map.data(), map_length, (MapT)0, nrows, stream);
    raft::update_host(h_map.data(), d_map.data(), map_length, stream);

    // expected and actual output matrix setup
    h_out.resize(map_length * ncols);
    d_out_exp.resize(map_length * ncols, stream);
    d_out_act.resize(map_length * ncols, stream);

    // launch gather on the host and copy the results to device
    naiveGather(h_in.data(), ncols, nrows, h_map.data(), map_length, h_out.data());
    raft::update_device(d_out_exp.data(), h_out.data(), map_length * ncols, stream);

    // launch device version of the kernel
    gatherLaunch(d_in.data(), ncols, nrows, d_map.data(), map_length, d_out_act.data(), stream);

    raft::interruptible::synchronize(stream);
  }
  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 protected:
  cudaStream_t stream = 0;
  GatherInputs params;
  std::vector<MatrixT> h_in, h_out;
  std::vector<MapT> h_map;
  rmm::device_uvector<MatrixT> d_in, d_out_exp, d_out_act;
  rmm::device_uvector<MapT> d_map;
};

const std::vector<GatherInputs> inputs = {{1024, 32, 128, 1234ULL},
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
                                          {1024, 128, 1024, 1234ULL}};

typedef GatherTest<float, uint32_t> GatherTestF;
TEST_P(GatherTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    d_out_exp.data(), d_out_act.data(), params.map_length * params.ncols, raft::Compare<float>()));
}

typedef GatherTest<double, uint32_t> GatherTestD;
TEST_P(GatherTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    d_out_exp.data(), d_out_act.data(), params.map_length * params.ncols, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestF, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestD, ::testing::ValuesIn(inputs));

}  // end namespace Matrix
}  // end namespace MLCommon
