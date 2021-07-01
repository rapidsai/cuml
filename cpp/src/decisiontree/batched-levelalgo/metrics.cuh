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

#include <cmath>
#include <common/grid_sync.cuh>
#include <cub/cub.cuh>
#include <limits>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

struct IntBin {
  int x;

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
  GiniObjectiveFunction() = default;
  GiniObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                        IdxT min_samples_leaf, double min, double max)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}

  DI IdxT NumClasses() const { return nclasses; }
  DI void IncrementHistogram(IntBin* hist, int nbins, int b, int label) {
    auto offset = label * nbins + b;
    IntBin::AtomicAdd(hist + offset, {1});
  }
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
  DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    int class_idx = 0;
    int count = 0;
    for (int i = 0; i < nclasses; i++) {
      auto current_count = shist[i].x;
      if (current_count > count) {
        class_idx = i;
        count = current_count;
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
  EntropyObjectiveFunction() = default;
  EntropyObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                           IdxT min_samples_leaf, double min, double max)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return nclasses; }
  DI void IncrementHistogram(IntBin* hist, int nbins, int b, int label) {
    auto offset = label * nbins + b;
    IntBin::AtomicAdd(hist + offset, {1});
  }
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
          auto total_sum = scdf_labels[nbins * j + nbins - 1].x;
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
  DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    int class_idx = 0;
    int count = 0;
    for (int i = 0; i < nclasses; i++) {
      auto current_count = shist[i].x;
      if (current_count > count) {
        class_idx = i;
        count = current_count;
      }
    }
    return class_idx;
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
  MSEObjectiveFunction() = default;
  HDI MSEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                           IdxT min_samples_leaf, double min, double max)
    : min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return 1; }
  DI void IncrementHistogram(MSEBin* hist, int nbins, int b, double label) {
    MSEBin::AtomicAdd(hist + b, {label, 1});
  }
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

  DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    return shist[0].label_sum / shist[0].count;
  }
};

template <int k_>
struct CosBin {
  static const int k = k_;
  double moments[k_ + 1];

  // For unit tests on the CPU
  void Add(double x, double min, double max) {
    double scaled_x = (x - min) * M_PI / (max - min);
    moments[0] += 1.0;
    for (int i = 1; i < k + 1; i++) {
      moments[i] += cos(i * scaled_x);
    }
  }

  HDI double NumItems() const { return moments[0]; }

  HDI double Cdf(double y, double min, double max) const {
    if ((max - min) == 0.0 || moments[0] == 0.0) return 0.5;
    double y_scaled = (y - min) * M_PI / (max - min);
    double sum = y_scaled / M_PI;
    double norm = 2.0 / (M_PI * moments[0]);
    for (int i = 1; i < k + 1; i++) {
      sum += moments[i] * norm * sin(i * y_scaled) / i;
    }
    return sum;
  }

  HDI double CdfIntegral(double z, double min, double max)const  {
    double a = min;
    double b = max;
    double L = b - a;
    if ((max - min) == 0.0 || moments[0] == 0.0) return 0.0;
    double z_scaled = (z - min) * M_PI / (max - min);
    double sum = z_scaled * z_scaled / (2.0 * M_PI);
    double norm = 2.0 / (M_PI * moments[0]);
    for (int i = 1; i < k + 1; i++) {
      sum += norm * (moments[i] - moments[i] * cos(i * z_scaled)) / (i * i);
    }
    return sum * L / M_PI;
  }

  HDI double AbsoluteError(double min, double max) const {
    double median = this->Median(min, max);
    double mae_l = this->CdfIntegral(median, min, max);
    double mae_r = mae_l - this->CdfIntegral(max, min, max);
    return (mae_l + mae_r + max - median) * this->NumItems();
  }

  // Bisection algorithm
  HDI double Median(double min, double max) const {
    const int iter = 20;
    double dx = max - min;
    double ymid = max;
    double fmid = 0.0;
    double rtb = min;
    for (int i = 0; i < iter; i++) {
      ymid = rtb + (dx *= 0.5);
      fmid = this->Cdf(ymid, min, max);
      if (fmid <= 0.5) rtb = ymid;
      if (fmid == 0.5) break;
    }
    return rtb;
  }

  DI static void AtomicAdd(CosBin* address, CosBin val) {
    for (int i = 0; i < k + 1; i++) {
      atomicAdd(&address->moments[i], val.moments[i]);
    }
  }
  HDI CosBin& operator+=(const CosBin& b) {
    for (int i = 0; i < k + 1; i++) {
      moments[i] += b.moments[i];
    }
    return *this;
  }
  HDI CosBin operator+(CosBin b) const {
    CosBin<k> tmp=*this;
    tmp  += b;
    return tmp;
  }
  HDI CosBin& operator-=(const CosBin& b) {
    for (int i = 0; i < k + 1; i++) {
      moments[i] -= b.moments[i];
    }
    return *this;
  }
  HDI CosBin operator-(CosBin b) const {
    CosBin<k> tmp=*this;
    tmp -= b;
    return tmp;
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class MAEObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;
  double min;
  double max;

 public:
  using BinT = CosBin<8>;
  MAEObjectiveFunction() = default;
  MAEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                       IdxT min_samples_leaf, double min, double max)
    : min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf),
      min(min),
      max(max) {}

  DI IdxT NumClasses() const { return 1; }
  DI void IncrementHistogram(BinT* bins, int nbins, int b, double label) {
    double scaled_label = (label - min) * M_PI / (max - min);
    atomicAdd(&(bins + b)->moments[0], 1.0);
    for (int i = 1; i < BinT::k + 1; i++) {
      atomicAdd(&(bins + b)->moments[i], cos(i * scaled_label));
    }
  }
  HDI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col,
                             IdxT len, IdxT nbins) {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      double gain = 0.0;
      const auto& left_bin = scdf_labels[i];
      const auto& parent_bin = scdf_labels[nbins - 1];
      const auto right_bin = parent_bin - left_bin;
      if (left_bin.NumItems() < min_samples_leaf ||
          right_bin.NumItems() < min_samples_leaf) {
        gain = -std::numeric_limits<DataT>::max();
      } else {
        gain = parent_bin.AbsoluteError(min, max) -
               (left_bin.AbsoluteError(min, max) +
                right_bin.AbsoluteError(min, max));
        gain /= parent_bin.NumItems();
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, left_bin.NumItems()});
    }
    return sp;
  }
  DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    return shist[0].Median(min, max);
  }
};

}  // namespace DecisionTree
}  // namespace ML
