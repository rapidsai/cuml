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

#pragma once

#include <common/grid_sync.h>
#include <cuda_utils.h>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * @brief Compute gain based on gini impurity metric
 * @param shist left/right class histograms for all bins (nbins x 2 x nclasses)
 * @param sbins quantiles for the current column (len = nbins)
 * @param parentGain parent node's best gain
 * @param sp will contain the per-thread best split so far
 * @param col current column
 * @param len total number of samples for the current node to be split
 * @param nbins number of bins
 * @param nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void giniGain(const int* shist, const DataT* sbins, DataT parentGain,
                 Split<DataT, IdxT>& sp, IdxT col, IdxT len, IdxT nbins,
                 IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[i * 2 * nclasses + j];
    }
    auto invLeft = One / nLeft;
    auto invRight = One / (len - nLeft);
    auto sum = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval = DataT(shist[i * 2 * nclasses + j]);
      sum += lval * invLeft * lval * invlen;
      auto rval = DataT(shist[i * 2 * nclasses + nclasses + j]);
      sum += rval * invRight * rval * invlen;
    }
    auto gain = parentGain - One + sum;
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on entropy
 * @param shist left/right class histograms for all bins (nbins x 2 x nclasses)
 * @param sbins quantiles for the current column (len = nbins)
 * @param parentGain parent node's best gain
 * @param sp will contain the per-thread best split so far
 * @param col current column
 * @param len total number of samples for the current node to be split
 * @param nbins number of bins
 * @param nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void entropyGain(const int* shist, const DataT* sbins, DataT parentGain,
                    Split<DataT, IdxT>& sp, IdxT col, IdxT len, IdxT nbins,
                    IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[i * 2 * nclasses + j];
    }
    auto invLeft = One / nLeft;
    auto invRight = One / (len - nLeft);
    auto sum = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lhistval = shist[i * 2 * nclasses + j];
      if (lhistval != 0) {
        auto lval = DataT(lhistval);
        sum += MLCommon::myLog(lval * invLeft) * lval * invlen;
      }
      auto rhistval = shist[i * 2 * nclasses + nclasses + j];
      if (rhistval != 0) {
        auto rval = DataT(rhistval);
        sum += MLCommon::myLog(rval * invRight) * rval * invlen;
      }
    }
    auto gain = parentGain + sum;
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @defgroup ClassificationMetrics Metric computation
 * @{
 * @brief Computes the initial metric on CPU
 * @tparam DataT data type
 * @tparam IdxT index type
 * @param h_hist class histogram
 * @param nclasses number of classes
 * @param nSampledRows number of rows upon which class histogram was computed
 * @return the computed metric
 */
template <typename DataT, typename IdxT>
DataT giniMetric(const int* h_hist, IdxT nclasses, IdxT nSampledRows) {
  const auto one = DataT(1.0);
  auto out = one;
  auto invlen = one / DataT(nSampledRows);
  for (IdxT i = 0; i < nclasses; ++i) {
    auto val = h_hist[i] * invlen;
    out -= val * val;
  }
  return out;
}

template <typename DataT, typename IdxT>
DataT entropyMetric(const int* h_hist, IdxT nclasses, IdxT nSampledRows) {
  const auto one = DataT(1.0);
  auto out = one;
  auto invlen = one / DataT(nSampledRows);
  for (IdxT i = 0; i < nclasses; ++i) {
    if (h_hist[i] != 0) {
      auto val = h_hist[i] * invlen;
      out -= MLCommon::myLog(val) * val;
    }
  }
  return out;
}
/** @} */

}  // namespace DecisionTree
}  // namespace ML
