/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/grid_sync.cuh>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * @brief Compute gain based on gini impurity metric
 *
 * @param[in]    shist    left/right class histograms for all bins
 *                        [dim = nbins x 2 x nclasses]
 * @param[in]    sbins    quantiles for the current column [len = nbins]
 * @param[inout] sp       will contain the per-thread best split so far
 * @param[in]    col      current column
 * @param[in]    len      total number of samples for the current node to be
 *                        split
 * @param[in]    nbins    number of bins
 * @param[in]    nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void giniGain(int* shist, DataT* sbins, Split<DataT, IdxT>& sp, IdxT col,
                 IdxT len, IdxT nbins, IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[i * 2 * nclasses + j];
    }
    auto nRight = len - nLeft;
    auto invLeft = One / nLeft;
    auto invRight = One / nRight;
    auto gain = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      int val_i = 0;
      if (nLeft != 0) {
        auto lval_i = shist[i * 2 * nclasses + j];
        auto lval = DataT(lval_i);
        gain += lval * invLeft * lval * invlen;
        val_i += lval_i;
      }
      if (nRight != 0) {
        auto rval_i = shist[i * 2 * nclasses + nclasses + j];
        auto rval = DataT(rval_i);
        gain += rval * invRight * rval * invlen;
        val_i += rval_i;
      }
      auto val = DataT(val_i) * invlen;
      gain -= val * val;
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on entropy
 *
 * @param[in]    shist    left/right class histograms for all bins
 *                        [dim = nbins x 2 x nclasses]
 * @param[in]    sbins    quantiles for the current column [len = nbins]
 * @param[inout] sp       will contain the per-thread best split so far
 * @param[in]    col      current column
 * @param[in]    len      total number of samples for the current node to be split
 * @param[in]    nbins    number of bins
 * @param[in]    nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void entropyGain(int* shist, DataT* sbins, Split<DataT, IdxT>& sp, IdxT col,
                    IdxT len, IdxT nbins, IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[i * 2 * nclasses + j];
    }
    auto nRight = len - nLeft;
    auto invLeft = One / nLeft;
    auto invRight = One / nRight;
    auto gain = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      int val_i = 0;
      if (nLeft != 0) {
        auto lval_i = shist[i * 2 * nclasses + j];
        if (lval_i != 0) {
          auto lval = DataT(lval_i);
          gain += raft::myLog(lval * invLeft) * lval * invlen;
        }
        val_i += lval_i;
      }
      if (nRight != 0) {
        auto rval_i = shist[i * 2 * nclasses + nclasses + j];
        if (rval_i != 0) {
          auto rval = DataT(rval_i);
          gain += raft::myLog(rval * invRight) * rval * invlen;
        }
        val_i += rval_i;
      }
      if (val_i != 0) {
        auto val = DataT(val_i) * invlen;
        gain -= val * raft::myLog(val);
      }
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on MSE
 *
 * @param[in]    spred  left/right child mean prediction for all bins
 *                      [dim = 2 x bins]
 * @param[in]    scount left child count for all bins [len = nbins]
 * @param[in]    sbins  quantiles for the current column [len = nbins]
 * @param[inout] sp     will contain the per-thread best split so far
 * @param[in]    col    current column
 * @param[in]    len    total number of samples for the current node to be split
 * @param[in]    nbins  number of bins
 */
template <typename DataT, typename IdxT>
DI void mseGain(DataT* spred, IdxT* scount, DataT* sbins,
                Split<DataT, IdxT>& sp, IdxT col, IdxT len, IdxT nbins) {
  auto invlen = DataT(1.0) / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto nLeft = scount[i];
    auto nRight = len - nLeft;
    auto invLeft = (DataT)len / nLeft;
    auto invRight = (DataT)len / nRight;
    auto valL = spred[i];
    auto valR = spred[nbins + i];
    // parent sum is basically sum of its left and right children
    auto valP = (valL + valR) * invlen;
    DataT gain = -valP * valP;
    if (nLeft != 0) {
      gain += valL * invlen * valL * invLeft;
    }
    if (nRight != 0) {
      gain += valR * invlen * valR * invRight;
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on MAE
 *
 * @param[in]    spred   left/right child sum of abs diff of prediction for all
 *                       bins [dim = 2 x bins]
 * @param[in]    spredP  parent's sum of abs diff of prediction for all bins
 *                       [dim = 2 x bins]
 * @param[in]    scount  left child count for all bins [len = nbins]
 * @param[in]    sbins   quantiles for the current column [len = nbins]
 * @param[inout] sp      will contain the per-thread best split so far
 * @param[in]    col     current column
 * @param[in]    len     total number of samples for current node to be split
 * @param[in]    nbins   number of bins
 */
template <typename DataT, typename IdxT>
DI void maeGain(DataT* spred, DataT* spredP, IdxT* scount, DataT* sbins,
                Split<DataT, IdxT>& sp, IdxT col, IdxT len, IdxT nbins) {
  auto invlen = DataT(1.0) / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto nLeft = scount[i];
    auto nRight = len - nLeft;
    DataT gain = spredP[i];
    if (nLeft != 0) {
      gain -= spred[i];
    }
    if (nRight != 0) {
      gain -= spred[i + nbins];
    }
    gain *= invlen;
    sp.update({sbins[i], col, gain, nLeft});
  }
}

}  // namespace DecisionTree
}  // namespace ML
