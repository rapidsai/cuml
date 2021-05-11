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

struct IntBin {
  int x;

  DI static void AtomicAdd(IntBin* hist, int nbins, int b, int label) {
    auto offset = label * (1 + nbins) + b;
    atomicAdd(&(hist + offset)->x, 1);
  }
  DI static void AtomicAddGlobal(IntBin* address, IntBin val) {
    atomicAdd(&address->x, val.x);
  }
  DI IntBin& operator+=(const IntBin& b) {
    x += b.x;
    return *this;
  }
  DI IntBin operator+(IntBin b) const {
    b += *this;
    return b;
  }
};

template <typename DataT, typename IdxT>
class GiniObjectiveFunction {
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = IntBin;
  GiniObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                        IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}

  DI IdxT NumClasses() const { return nclasses; }
  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col,
                             IdxT len, IdxT nbins) {
    Split<DataT, IdxT> sp;
    constexpr DataT One = DataT(1.0);
    DataT invlen = One / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      int nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[nbins * j + i].x;
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
          auto lval_i = scdf_labels[nbins * j + i].x;
          auto lval = DataT(lval_i);
          gain += lval * invLeft * lval * invlen;

          val_i += lval_i;
          auto total_sum = scdf_labels[nbins * j + nbins - 1].x;
          auto rval_i = total_sum - lval_i;
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
    return sp;
  }
};

template <typename DataT, typename IdxT>
class EntropyObjectiveFunction {
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = IntBin;
  EntropyObjectiveFunction(DataT nclasses, IdxT min_impurity_decrease,
                           IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return nclasses; }
  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col,
                             IdxT len, IdxT nbins) {
    Split<DataT, IdxT> sp;
    constexpr DataT One = DataT(1.0);
    DataT invlen = One / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      int nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[nbins * j + i].x;
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
          auto lval_i = scdf_labels[nbins * j + i].x;
          if (lval_i != 0) {
            auto lval = DataT(lval_i);
            gain += raft::myLog(lval * invLeft) / raft::myLog(DataT(2)) * lval *
                    invlen;
          }

          val_i += lval_i;
          auto total_sum = scdf_labels[2 * nbins * j + nbins - 1].x;
          auto rval_i = total_sum - lval_i;
          if (rval_i != 0) {
            auto rval = DataT(rval_i);
            gain += raft::myLog(rval * invRight) / raft::myLog(DataT(2)) *
                    rval * invlen;
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
    return sp;
  }
};

template <typename DataT, typename IdxT>
class MSEObjectiveFunction {
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  HDI MSEObjectiveFunction(DataT min_impurity_decrease, IdxT min_samples_leaf)
    : min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return 1; }
  DI Split<DataT, IdxT> Gain(DataT* slabel_cdf, IdxT* scount_cdf,
                             DataT label_sum, DataT* sbins, IdxT col, IdxT len,
                             IdxT nbins) {
    Split<DataT, IdxT> sp;
    auto invlen = DataT(1.0) / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = scount_cdf[i];
      auto nRight = len - nLeft;
      DataT gain;
      // if there aren't enough samples in this split, don't bother!
      if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
        gain = -NumericLimits<DataT>::kMax;
      } else {
        DataT parent_obj = -label_sum * label_sum / len;
        DataT left_obj = -(slabel_cdf[i] * slabel_cdf[i]) / nLeft;
        DataT right_label_sum = slabel_cdf[i] - label_sum;
        DataT right_obj = -(right_label_sum * right_label_sum) / nRight;
        gain = parent_obj - (left_obj + right_obj);
        gain *= invlen;
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -NumericLimits<DataT>::kMax;
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
};

}  // namespace DecisionTree
}  // namespace ML
