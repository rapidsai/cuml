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
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"
#include <limits>

namespace ML {
namespace DecisionTree {

struct IntBin {
  int x;

  DI static void IncrementHistogram(IntBin* hist, int nbins, int b, int label) {
    auto offset = label * nbins + b;
    IntBin::AtomicAdd(hist + offset, {1});
  }
  DI static void AtomicAdd(IntBin* address, IntBin val) {
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

template <typename DataT_, typename LabelT_, typename IdxT_>
class GiniObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;
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
        gain = -std::numeric_limits<DataT>::max();
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
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
  static DI LabelT LeafPrediction(BinT* shist, int nclasses, DataT* aux) {
    int class_idx = 0;
    int count = 0;
    int total_count = 0;
    for (int i = 0; i < nclasses; i++) {
      auto current_count = shist[i].x;
      total_count += current_count;
      if (current_count > count) {
        class_idx = i;
        count = current_count;
      }
    }
    if (aux) {
      if (nclasses == 2) {
        // Special handling for binary classifiers
        *aux = static_cast<DataT>(shist[1].x) / total_count;
      } else {
        *aux = DataT(0);
      }
    }
    return class_idx;
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class EntropyObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;
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
        gain = -std::numeric_limits<DataT>::max();
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
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
  static DI LabelT LeafPrediction(BinT* shist, int nclasses, DataT* aux) {
    // Same as Gini
    return GiniObjectiveFunction<DataT, LabelT, IdxT>::LeafPrediction(
      shist, nclasses, aux);
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class MSEObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;

 private:
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  struct MSEBin {
    double label_sum;
    int count;

    DI static void IncrementHistogram(MSEBin* hist, int nbins, int b,
                                      double label) {
      MSEBin::AtomicAdd(hist + b, {label, 1});
    }
    DI static void AtomicAdd(MSEBin* address, MSEBin val) {
      atomicAdd(&address->label_sum, val.label_sum);
      atomicAdd(&address->count, val.count);
    }
    DI MSEBin& operator+=(const MSEBin& b) {
      label_sum += b.label_sum;
      count += b.count;
      return *this;
    }
    DI MSEBin operator+(MSEBin b) const {
      b += *this;
      return b;
    }
  };
  using BinT = MSEBin;
  HDI MSEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                           IdxT min_samples_leaf)
    : min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return 1; }
  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* sbins, IdxT col, IdxT len,
                             IdxT nbins) {
    Split<DataT, IdxT> sp;
    auto invlen = DataT(1.0) / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      auto nRight = len - nLeft;
      DataT gain;
      // if there aren't enough samples in this split, don't bother!
      if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
        gain = -std::numeric_limits<DataT>::max();
      } else {
        auto label_sum = shist[nbins - 1].label_sum;
        DataT parent_obj = -label_sum * label_sum / len;
        DataT left_obj = -(shist[i].label_sum * shist[i].label_sum) / nLeft;
        DataT right_label_sum = shist[i].label_sum - label_sum;
        DataT right_obj = -(right_label_sum * right_label_sum) / nRight;
        gain = parent_obj - (left_obj + right_obj);
        gain *= invlen;
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT* shist, int nclasses, DataT* aux) {
    return shist[0].label_sum / shist[0].count;
  }
};

}  // namespace DecisionTree
}  // namespace ML
