/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace {

template <typename DataT>
class NumericLimits;

template <>
class NumericLimits<float> {
 public:
  static constexpr double kMax = __FLT_MAX__;
};

template <>
class NumericLimits<double> {
 public:
  static constexpr double kMax = __DBL_MAX__;
};

}  // anonymous namespace

namespace ML {
namespace DecisionTree {

/**
 * @brief Compute gain based on gini impurity metric
 *
 * @param[in]    shist                 left/right class histograms for all bins
 *                                     [dim = nbins x 2 x nclasses]
 * @param[in]    sbins                 quantiles for the current column
 *                                     [len = nbins]
 * @param[inout] sp                    will contain the per-thread best split
 *                                     so far
 * @param[in]    col                   current column
 * @param[in]    len                   total number of samples for the current
 *                                     node to be split
 * @param[in]    nbins                 number of bins
 * @param[in]    nclasses              number of classes
 * @param[in]    min_samples_leaf      minimum number of samples per each leaf.
 *                                     Any splits that lead to a leaf node with
 *                                     samples fewer than min_samples_leaf will
 *                                     be ignored.
 * @param[in]    min_impurity_decrease minimum improvement in MSE metric. Any
 *                                     splits that do not improve (decrease)
 *                                     the MSE metric at least by this amount
 *                                     will be ignored.
 */
template <typename DataT, typename IdxT>
DI void giniGain(int* shist, DataT* sbins, Split<DataT, IdxT>& sp, IdxT col,
                 IdxT len, IdxT nbins, IdxT nclasses, IdxT min_samples_leaf,
                 DataT min_impurity_decrease) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[2 * nbins * j + i];
    }
    auto nRight = len - nLeft;
    auto gain = DataT(0.0);
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      gain = -NumericLimits<DataT>::kMax;
    } else {
      auto invLeft = One / nLeft;
      auto invRight = One / nRight;
      for (IdxT j = 0; j < nclasses; ++j) {
        int val_i = 0;
        auto lval_i = shist[2 * nbins * j + i];
        auto lval = DataT(lval_i);
        gain += lval * invLeft * lval * invlen;

        val_i += lval_i;
        auto rval_i = shist[2 * nbins * j + nbins + i];
        auto rval = DataT(rval_i);
        gain += rval * invRight * rval * invlen;

        val_i += rval_i;
        auto val = DataT(val_i) * invlen;
        gain -= val * val;
      }
    }
    // if the gain is not "enough", don't bother!
    if (gain <= min_impurity_decrease) {
      gain = -NumericLimits<DataT>::kMax;
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on entropy
 *
 * @param[in]    shist                 left/right class histograms for all bins
 *                                     [dim = nbins x 2 x nclasses]
 * @param[in]    sbins                 quantiles for the current column
 *                                     [len = nbins]
 * @param[inout] sp                    will contain the per-thread best split
 *                                     so far
 * @param[in]    col                   current column
 * @param[in]    len                   total number of samples for the current
 *                                     node to be split
 * @param[in]    nbins                 number of bins
 * @param[in]    nclasses              number of classes
 * @param[in]    min_samples_leaf      minimum number of samples per each leaf.
 *                                     Any splits that lead to a leaf node with
 *                                     samples fewer than min_samples_leaf will
 *                                     be ignored.
 * @param[in]    min_impurity_decrease minimum improvement in MSE metric. Any
 *                                     splits that do not improve (decrease)
 *                                     the MSE metric at least by this amount
 *                                     will be ignored.
 */
template <typename DataT, typename IdxT>
DI void entropyGain(int* shist, DataT* sbins, Split<DataT, IdxT>& sp, IdxT col,
                    IdxT len, IdxT nbins, IdxT nclasses, IdxT min_samples_leaf,
                    DataT min_impurity_decrease) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    int nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += shist[2 * nbins * j + i];
    }
    auto nRight = len - nLeft;
    auto gain = DataT(0.0);
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      gain = -NumericLimits<DataT>::kMax;
    } else {
      auto invLeft = One / nLeft;
      auto invRight = One / nRight;
      for (IdxT j = 0; j < nclasses; ++j) {
        int val_i = 0;
        auto lval_i = shist[2 * nbins * j + i];
        if (lval_i != 0) {
          auto lval = DataT(lval_i);
          gain +=
            raft::myLog(lval * invLeft) / raft::myLog(DataT(2)) * lval * invlen;
        }

        val_i += lval_i;
        auto rval_i = shist[2 * nbins * j + nbins + i];
        if (rval_i != 0) {
          auto rval = DataT(rval_i);
          gain += raft::myLog(rval * invRight) / raft::myLog(DataT(2)) * rval *
                  invlen;
        }

        val_i += rval_i;
        if (val_i != 0) {
          auto val = DataT(val_i) * invlen;
          gain -= val * raft::myLog(val) / raft::myLog(DataT(2));
        }
      }
    }
    // if the gain is not "enough", don't bother!
    if (gain <= min_impurity_decrease) {
      gain = -NumericLimits<DataT>::kMax;
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

/**
 * @brief Compute gain based on MSE or MAE
 *
 * @param[in]    spred                 left/right child sum of abs diff of
 *                                     prediction for all bins [dim = 2 x bins]
 * @param[in]    spredP                parent's sum of abs diff of prediction
 *                                     for all bins [dim = 2 x bins]
 * @param[in]    scount                left child count for all bins
 *                                     [len = nbins]
 * @param[in]    sbins                 quantiles for the current column
 *                                     [len = nbins]
 * @param[inout] sp                    will contain the per-thread best split
 *                                     so far
 * @param[in]    col                   current column
 * @param[in]    len                   total number of samples for current node
 *                                     to be split
 * @param[in]    nbins                 number of bins
 * @param[in]    min_samples_leaf      minimum number of samples per each leaf.
 *                                     Any splits that lead to a leaf node with
 *                                     samples fewer than min_samples_leaf will
 *                                     be ignored.
 * @param[in]    min_impurity_decrease minimum improvement in MSE metric. Any
 *                                     splits that do not improve (decrease)
 *                                     the MSE metric at least by this amount
 *                                     will be ignored.
 */
template <typename DataT, typename IdxT>
DI void regressionMetricGain(DataT* spred, DataT* spredP, IdxT* scount,
                             DataT* sbins, Split<DataT, IdxT>& sp, IdxT col,
                             IdxT len, IdxT nbins, IdxT min_samples_leaf,
                             DataT min_impurity_decrease) {
  auto invlen = DataT(1.0) / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto nLeft = scount[i];
    auto nRight = len - nLeft;
    DataT gain;
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      gain = -NumericLimits<DataT>::kMax;
    } else {
      gain = spredP[i] - spred[i] - spred[i + nbins];
      gain *= invlen;
    }
    // if the gain is not "enough", don't bother!
    if (gain <= min_impurity_decrease) {
      gain = -NumericLimits<DataT>::kMax;
    }
    sp.update({sbins[i], col, gain, nLeft});
  }
}

}  // namespace DecisionTree
}  // namespace ML
