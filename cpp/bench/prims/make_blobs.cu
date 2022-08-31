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

#include <common/ml_benchmark.hpp>
#include <raft/random/make_blobs.hpp>

namespace MLCommon {
namespace Bench {
namespace Random {

struct Params {
  int rows, cols, clusters;
  bool row_major;
};  // struct Params

template <typename T>
struct MakeBlobs : public Fixture {
  MakeBlobs(const std::string& name, const Params& p)
    : Fixture(name), params(p), data(0, stream), labels(0, stream)
  {
  }

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    data.resize(params.rows * params.cols, stream);
    labels.resize(params.rows, stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    data.release();
    labels.release();
  }
  void runBenchmark(::benchmark::State& state) override
  {
    loopOnState(state, [this]() {
      raft::random::make_blobs(data.data(),
                               labels.data(),
                               params.rows,
                               params.cols,
                               params.clusters,
                               this->stream,
                               params.row_major);
    });
  }

 private:
  Params params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<int> labels;
};  // struct MakeBlobs

static std::vector<Params> getInputs()
{
  std::vector<Params> out;
  Params p;
  for (auto rows : std::vector<int>{100000, 1000000}) {
    for (auto cols : std::vector<int>{10, 100}) {
      for (auto clusters : std::vector<int>{2, 10, 100}) {
        p.rows      = rows;
        p.cols      = cols;
        p.clusters  = clusters;
        p.row_major = true;
        out.push_back(p);
        p.row_major = false;
        out.push_back(p);
      }
    }
  }
  return out;
}

ML_BENCH_REGISTER(Params, MakeBlobs<float>, "", getInputs());
ML_BENCH_REGISTER(Params, MakeBlobs<double>, "", getInputs());

}  // namespace Random
}  // namespace Bench
}  // namespace MLCommon
