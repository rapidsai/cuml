/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#pragma once

#include "adjgraph/runner.h"
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "common/nvtx.hpp"
#include "cuda_utils.h"
#include "label/classlabels.h"
#include "sparse/csr.h"
#include "vertexdeg/runner.h"

namespace Dbscan {

using namespace MLCommon;

static const int TPB = 256;

/**
 * Adjust labels from weak_cc primitive to match sklearn:
 * 1. Turn any labels matching MAX_LABEL into -1
 * 2. Subtract 1 from all other labels.
 */
template <typename Type, typename Index_ = int>
__global__ void relabelForSkl(Type* labels, Index_ N, Type MAX_LABEL) {
  Index_ tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (labels[tid] == MAX_LABEL)
    labels[tid] = -1;
  else if (tid < N)
    --labels[tid];
}

/**
 * Turn the non-monotonic labels from weak_cc primitive into
 * an array of labels drawn from a monotonically increasing set.
 */
template <typename Type, typename Index_ = int>
void final_relabel(Type* db_cluster, Index_ N, cudaStream_t stream) {
  Type MAX_LABEL = std::numeric_limits<Type>::max();
  MLCommon::Label::make_monotonic(
    db_cluster, db_cluster, N, stream,
    [MAX_LABEL] __device__(int val) { return val == MAX_LABEL; });
}

/* @param N number of points
 * @param D dimensionality of the points
 * @param eps epsilon neighborhood criterion
 * @param minPts core points criterion
 * @param labels the output labels (should be of size N)
 * @param ....
 * @param temp temporary global memory buffer used to store intermediate computations
 *             If this is a null pointer, then this function will return the workspace size needed.
 *             It is the responsibility of the user to cudaMalloc and cudaFree this buffer!
 * @param stream the cudaStream where to launch the kernels
 * @return in case the temp buffer is null, this returns the size needed.
 */
template <typename Type, typename Type_f, typename Index_ = int>
size_t run(const ML::cumlHandle_impl& handle, Type_f* x, Index_ N, Index_ D,
           Type_f eps, Type minPts, Type* labels, int algoVd, int algoAdj,
           int algoCcl, void* workspace, Index_ nBatches, cudaStream_t stream) {
  const size_t align = 256;
  Index_ batchSize = ceildiv(N, nBatches);
  size_t adjSize = alignTo<size_t>(sizeof(bool) * N * batchSize, align);
  size_t corePtsSize = alignTo<size_t>(sizeof(bool) * batchSize, align);
  size_t xaSize = alignTo<size_t>(sizeof(bool) * N, align);
  size_t mSize = alignTo<size_t>(sizeof(bool), align);
  size_t vdSize = alignTo<size_t>(sizeof(int) * (batchSize + 1), align);
  size_t exScanSize = alignTo<size_t>(sizeof(Type) * batchSize, align);

  if (workspace == NULL) {
    auto size =
      adjSize + corePtsSize + 2 * xaSize + mSize + vdSize + exScanSize;
    return size;
  }
  // partition the temporary workspace needed for different stages of dbscan
  Type adjlen = 0;
  Type curradjlen = 0;
  char* temp = (char*)workspace;
  bool* adj = (bool*)temp;
  temp += adjSize;
  bool* core_pts = (bool*)temp;
  temp += corePtsSize;
  bool* xa = (bool*)temp;
  temp += xaSize;
  bool* fa = (bool*)temp;
  temp += xaSize;
  bool* m = (bool*)temp;
  temp += mSize;
  int* vd = (int*)temp;
  temp += vdSize;
  Type* ex_scan = (Type*)temp;
  temp += exScanSize;

  // Running VertexDeg
  MLCommon::Sparse::WeakCCState<Type> state(xa, fa, m);

  for (int i = 0; i < nBatches; i++) {
    ML::PUSH_RANGE("Profile::Dbscan::VertexDeg");
    MLCommon::device_buffer<Type> adj_graph(handle.getDeviceAllocator(),
                                            stream);
    Type startVertexId = i * batchSize;
    int nPoints = min(N - startVertexId, batchSize);
    if (nPoints <= 0) continue;
    VertexDeg::run<Type_f, Index_>(handle, adj, vd, x, eps, N, D, algoVd,
                                   startVertexId, nPoints, stream);
    MLCommon::updateHost(&curradjlen, vd + nPoints, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ML::POP_RANGE();

    // Running AdjGraph
    ML::PUSH_RANGE("Profile::Dbscan::AdjGraph");
    if (curradjlen > adjlen || adj_graph.data() == NULL) {
      adjlen = curradjlen;
      adj_graph.resize(adjlen, stream);
    }
    AdjGraph::run<Type, Index_>(handle, adj, vd, adj_graph.data(), adjlen,
                                ex_scan, N, minPts, core_pts, algoAdj, nPoints,
                                stream);
    ML::POP_RANGE();

    ML::PUSH_RANGE("Profile::Dbscan::WeakCC");
    MLCommon::Sparse::weak_cc_batched<Type, TPB>(
      labels, ex_scan, adj_graph.data(), adjlen, N, startVertexId, batchSize,
      &state, stream,
      [core_pts] __device__(Type tid) { return core_pts[tid]; });
    ML::POP_RANGE();
  }

  ML::PUSH_RANGE("Profile::Dbscan::FinalRelabel");
  if (algoCcl == 2) final_relabel(labels, N, stream);
  Type MAX_LABEL = std::numeric_limits<Type>::max();
  int nblks = ceildiv<int>(N, TPB);
  relabelForSkl<Type><<<nblks, TPB, 0, stream>>>(labels, N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
  ML::POP_RANGE();
  return (size_t)0;
}
}  // namespace Dbscan
