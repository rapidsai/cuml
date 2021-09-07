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
#include <limits>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "split.cuh"

namespace ML {
namespace DT {

struct CountBin {
  int x;
  CountBin(CountBin const&) = default;
  HDI CountBin(int x_) : x(x_) {}
  HDI CountBin() : x(0){};

  DI static void IncrementHistogram(CountBin* hist, int nbins, int b, int label)
  {
    auto offset = label * nbins + b;
    CountBin::AtomicAdd(hist + offset, {1});
  }
  DI static void AtomicAdd(CountBin* address, CountBin val) { atomicAdd(&address->x, val.x); }
  HDI CountBin& operator+=(const CountBin& b)
  {
    x += b.x;
    return *this;
  }
  HDI CountBin operator+(CountBin b) const
  {
    b += *this;
    return b;
  }
};

struct AggregateBin {
  double label_sum;
  int count;

  AggregateBin(AggregateBin const&) = default;
  HDI AggregateBin() : label_sum(0.0), count(0){};
  HDI AggregateBin(double label_sum, int count) : label_sum(label_sum), count(count) {}

  DI static void IncrementHistogram(AggregateBin* hist, int nbins, int b, double label)
  {
    AggregateBin::AtomicAdd(hist + b, {label, 1});
  }
  DI static void AtomicAdd(AggregateBin* address, AggregateBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->count, val.count);
  }
  HDI AggregateBin& operator+=(const AggregateBin& b)
  {
    label_sum += b.label_sum;
    count += b.count;
    return *this;
  }
  HDI AggregateBin operator+(AggregateBin b) const
  {
    b += *this;
    return b;
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class GiniObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = CountBin;
  GiniObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease, IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf)
  {
  }

  DI IdxT NumClasses() const { return nclasses; }

  HDI DataT GainPerSplit(BinT* hist, IdxT i, IdxT nbins, IdxT len, IdxT nLeft)
  {
    auto nRight         = len - nLeft;
    constexpr DataT One = DataT(1.0);
    auto invlen         = One / len;
    auto invLeft        = One / nLeft;
    auto invRight       = One / nRight;
    auto gain           = DataT(0.0);

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    for (IdxT j = 0; j < nclasses; ++j) {
      int val_i   = 0;
      auto lval_i = hist[nbins * j + i].x;
      auto lval   = DataT(lval_i);
      gain += lval * invLeft * lval * invlen;

      val_i += lval_i;
      auto total_sum = hist[nbins * j + nbins - 1].x;
      auto rval_i    = total_sum - lval_i;
      auto rval      = DataT(rval_i);
      gain += rval * invRight * rval * invlen;

      val_i += rval_i;
      auto val = DataT(val_i) * invlen;
      gain -= val * val;
    }

    // if the gain is not "enough", don't bother!
    if (gain <= min_impurity_decrease)
      return -std::numeric_limits<DataT>::max();

    else
      return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* sbins, IdxT col, IdxT len, IdxT nbins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = IdxT(0);
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += shist[nbins * j + i].x;
      }
      sp.update({sbins[i], col, GainPerSplit(shist, i, nbins, len, nLeft), nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT const* shist, int nclasses)
  {
    int class_idx = 0;
    int count     = 0;
    for (int i = 0; i < nclasses; i++) {
      auto current_count = shist[i].x;
      if (current_count > count) {
        class_idx = i;
        count     = current_count;
      }
    }
    return class_idx;
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class EntropyObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = CountBin;
  EntropyObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease, IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf)
  {
  }
  DI IdxT NumClasses() const { return nclasses; }

  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT nbins, IdxT len, IdxT nLeft)
  {
    auto nRight{len - nLeft};
    auto gain{DataT(0.0)};
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      auto invLeft{DataT(1.0) / nLeft};
      auto invRight{DataT(1.0) / nRight};
      auto invLen{DataT(1.0) / len};
      for (IdxT c = 0; c < nclasses; ++c) {
        int val_i   = 0;
        auto lval_i = hist[nbins * c + i].x;
        if (lval_i != 0) {
          auto lval = DataT(lval_i);
          gain += raft::myLog(lval * invLeft) / raft::myLog(DataT(2)) * lval * invLen;
        }

        val_i += lval_i;
        auto total_sum = hist[nbins * c + nbins - 1].x;
        auto rval_i    = total_sum - lval_i;
        if (rval_i != 0) {
          auto rval = DataT(rval_i);
          gain += raft::myLog(rval * invRight) / raft::myLog(DataT(2)) * rval * invLen;
        }

        val_i += rval_i;
        if (val_i != 0) {
          auto val = DataT(val_i) * invLen;
          gain -= val * raft::myLog(val) / raft::myLog(DataT(2));
        }
      }

      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) return -std::numeric_limits<DataT>::max();

      return gain;
    }
  }

  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col, IdxT len, IdxT nbins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft{IdxT(0)};
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[nbins * j + i].x;
      }
      sp.update({sbins[i], col, GainPerSplit(scdf_labels, i, nbins, len, nLeft), nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT const* shist, int nclasses)
  {
    // Same as Gini
    return GiniObjectiveFunction<DataT, LabelT, IdxT>::LeafPrediction(shist, nclasses);
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class PoissonObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT                 = AggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

  HDI PoissonObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease, IdxT min_samples_leaf)
    : min_impurity_decrease(min_impurity_decrease), min_samples_leaf(min_samples_leaf)
  {
  }
  DI IdxT NumClasses() const { return 1; }

  /**
   * @brief compute the poisson impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy poisson half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children.
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT nbins, IdxT len, IdxT nLeft)
  {
    // get the lens'
    auto nRight = len - nLeft;

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[nbins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[nbins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = -label_sum * raft::myLog(label_sum / len);
    DataT left_obj   = -left_label_sum * raft::myLog(left_label_sum / nLeft);
    DataT right_obj  = -right_label_sum * raft::myLog(right_label_sum / nRight);
    auto gain        = parent_obj - (left_obj + right_obj);
    gain             = gain / len;

    // if the gain is not "enough", don't bother!
    if (gain <= min_impurity_decrease)
      return -std::numeric_limits<DataT>::max();

    else
      return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist, DataT const* sbins, IdxT col, IdxT len, IdxT nbins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({sbins[i], col, GainPerSplit(shist, i, nbins, len, nLeft), nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT const* shist, int nclasses)
  {
    return shist[0].label_sum / shist[0].count;
  }
};
template <typename DataT_, typename LabelT_, typename IdxT_>
class MSEObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = AggregateBin;
  HDI MSEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease, IdxT min_samples_leaf)
    : min_impurity_decrease(min_impurity_decrease), min_samples_leaf(min_samples_leaf)
  {
  }
  DI IdxT NumClasses() const { return 1; }

  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT nbins, IdxT len, IdxT nLeft)
  {
    auto gain{DataT(0)};
    auto nRight{len - nLeft};
    auto invLen{DataT(1.0) / len};
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      auto label_sum       = hist[nbins - 1].label_sum;
      auto parent_obj      = -label_sum * label_sum * invLen;
      auto left_obj        = -(hist[i].label_sum * hist[i].label_sum) / nLeft;
      auto right_label_sum = hist[i].label_sum - label_sum;
      auto right_obj       = -(right_label_sum * right_label_sum) / nRight;
      gain                 = parent_obj - (left_obj + right_obj);
      gain *= invLen;

      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) return -std::numeric_limits<DataT>::max();

      return gain;
    }
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist, DataT const* sbins, IdxT col, IdxT len, IdxT nbins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({sbins[i], col, GainPerSplit(shist, i, nbins, len, nLeft), nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT const* shist, int nclasses)
  {
    return shist[0].label_sum / shist[0].count;
  }
};

}  // namespace DT
}  // namespace ML
