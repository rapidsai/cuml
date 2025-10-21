/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "benchmark.cuh"

#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/util/cuda_utils.cuh>

#include <rmm/device_buffer.hpp>

#include <memory>
#include <utility>

namespace ML {
namespace Bench {
namespace umap {

struct Params {
  DatasetParams data;
  BlobsParams blobs;
  UMAPParams umap;
};

template <typename OutT, typename InT, typename IdxT>
CUML_KERNEL void castKernel(OutT* out, const InT* in, IdxT len)
{
  auto tid = IdxT(blockIdx.x) * blockDim.x + IdxT(threadIdx.x);
  if (tid < len) { out[tid] = OutT(in[tid]); }
}
template <typename OutT, typename InT, typename IdxT = int>
void cast(OutT* out, const InT* in, IdxT len, cudaStream_t stream)
{
  static const int TPB = 256;
  auto nblks           = raft::ceildiv<IdxT>(len, TPB);
  castKernel<OutT, InT, IdxT><<<nblks, TPB, 0, stream>>>(out, in, len);
  RAFT_CUDA_TRY(cudaGetLastError());
}

class UmapBase : public BlobsFixture<float, int> {
 public:
  UmapBase(const std::string& name, const Params& p)
    : BlobsFixture<float, int>(name, p.data, p.blobs), uParams(p.umap)
  {
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    using MLCommon::Bench::CudaEventTimer;
    if (!this->params.rowMajor) { state.SkipWithError("Umap only supports row-major inputs"); }
    this->loopOnState(state, [this]() { coreBenchmarkMethod(); });
  }

  virtual void coreBenchmarkMethod() = 0;

  void allocateTempBuffers(const ::benchmark::State& state) override
  {
    alloc(yFloat, this->params.nrows);
    cast<float, int>(yFloat, this->data.y.data(), this->params.nrows, this->stream);
  }

  void deallocateTempBuffers(const ::benchmark::State& state) override
  {
    dealloc(yFloat, this->params.nrows);
    embeddings_buffer.reset();
  }

  UMAPParams uParams;
  float* yFloat;
  std::unique_ptr<rmm::device_buffer> embeddings_buffer;
};  // class UmapBase

std::vector<Params> getInputs()
{
  std::vector<Params> out;
  Params p;
  p.data.rowMajor                          = true;
  p.blobs.cluster_std                      = 1.0;
  p.blobs.shuffle                          = false;
  p.blobs.center_box_min                   = -10.0;
  p.blobs.center_box_max                   = 10.0;
  p.blobs.seed                             = 12345ULL;
  p.umap.n_components                      = 4;
  p.umap.n_epochs                          = 500;
  p.umap.min_dist                          = 0.9f;
  std::vector<std::pair<int, int>> rowcols = {
    {10000, 500},
    {20000, 500},
    {40000, 500},
  };
  for (auto& rc : rowcols) {
    p.data.nrows = rc.first;
    p.data.ncols = rc.second;
    for (auto& nc : std::vector<int>({2, 10})) {
      p.data.nclasses = nc;
      out.push_back(p);
    }
  }
  return out;
}

class UmapSupervised : public UmapBase {
 public:
  UmapSupervised(const std::string& name, const Params& p) : UmapBase(name, p) {}

 protected:
  void coreBenchmarkMethod()
  {
    auto graph = raft::make_host_coo_matrix<float, int, int, uint64_t>(
      *this->handle, this->params.nrows, this->params.nrows);
    UMAP::fit(*this->handle,
              this->data.X.data(),
              yFloat,
              this->params.nrows,
              this->params.ncols,
              nullptr,
              nullptr,
              &uParams,
              embeddings_buffer,
              graph);
  }
};
ML_BENCH_REGISTER(Params, UmapSupervised, "blobs", getInputs());

class UmapUnsupervised : public UmapBase {
 public:
  UmapUnsupervised(const std::string& name, const Params& p) : UmapBase(name, p) {}

 protected:
  void coreBenchmarkMethod()
  {
    auto graph = raft::make_host_coo_matrix<float, int, int, uint64_t>(
      *this->handle, this->params.nrows, this->params.nrows);
    UMAP::fit(*this->handle,
              this->data.X.data(),
              nullptr,
              this->params.nrows,
              this->params.ncols,
              nullptr,
              nullptr,
              &uParams,
              embeddings_buffer,
              graph);
  }
};
ML_BENCH_REGISTER(Params, UmapUnsupervised, "blobs", getInputs());

class UmapTransform : public UmapBase {
 public:
  UmapTransform(const std::string& name, const Params& p) : UmapBase(name, p) {}

 protected:
  void coreBenchmarkMethod()
  {
    // Extract the embeddings pointer from the device_buffer
    float* embeddings_ptr = static_cast<float*>(embeddings_buffer->data());
    UMAP::transform(*this->handle,
                    this->data.X.data(),
                    this->params.nrows,
                    this->params.ncols,
                    this->data.X.data(),
                    this->params.nrows,
                    embeddings_ptr,
                    this->params.nrows,
                    &uParams,
                    transformed);
  }
  void allocateBuffers(const ::benchmark::State& state)
  {
    UmapBase::allocateBuffers(state);
    auto& handle = *this->handle;
    alloc(transformed, this->params.nrows * uParams.n_components);
    auto graph = raft::make_host_coo_matrix<float, int, int, uint64_t>(
      handle, this->params.nrows, this->params.nrows);
    UMAP::fit(handle,
              this->data.X.data(),
              yFloat,
              this->params.nrows,
              this->params.ncols,
              nullptr,
              nullptr,
              &uParams,
              embeddings_buffer,
              graph);
  }
  void deallocateBuffers(const ::benchmark::State& state)
  {
    dealloc(transformed, this->params.nrows * uParams.n_components);
    UmapBase::deallocateBuffers(state);
  }

 private:
  float* transformed;
};
ML_BENCH_REGISTER(Params, UmapTransform, "blobs", getInputs());

}  // end namespace umap
}  // end namespace Bench
}  // end namespace ML
