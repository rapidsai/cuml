/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bins.cuh"
#include "dataset.h"
#include "split.cuh"

#include <cuml/tree/algo_helper.h>

#include <limits>

namespace ML {
namespace DT {

template <typename DataT_, typename LabelT_, typename IdxT_>
class ClassificationObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  using BinT   = CountBin;
  using CountT = unsigned long long int;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf;
  CRITERION criterion;

  DI CountT CountLeft(BinT const* hist, IdxT i, IdxT n_bins) const
  {
    CountT nLeft = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += hist[n_bins * j + i].x;
    }
    return nLeft;
  }

  HDI DataT
  GiniGain(BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    constexpr DataT One = DataT(1.0);
    auto invLen         = One / DataT(len);
    auto invLeft        = One / DataT(nLeft);
    auto invRight       = One / DataT(nRight);
    auto gain           = DataT(0.0);

    for (IdxT j = 0; j < nclasses; ++j) {
      double val_i = 0.0;
      auto lval_i  = hist[n_bins * j + i].x;
      auto lval    = DataT(lval_i);
      gain += lval * invLeft * lval * invLen;

      val_i += lval_i;
      auto total_sum = hist[n_bins * j + n_bins - 1].x;
      auto rval_i    = total_sum - lval_i;
      auto rval      = DataT(rval_i);
      gain += rval * invRight * rval * invLen;

      val_i += rval_i;
      auto val = DataT(val_i) * invLen;
      gain -= val * val;
    }

    return gain;
  }

  HDI DataT
  EntropyGain(
    BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    auto gain{DataT(0.0)};
    auto invLeft{DataT(1.0) / DataT(nLeft)};
    auto invRight{DataT(1.0) / DataT(nRight)};
    auto invLen{DataT(1.0) / DataT(len)};
    for (IdxT c = 0; c < nclasses; ++c) {
      double val_i = 0.0;
      auto lval_i  = hist[n_bins * c + i].x;
      if (lval_i != 0) {
        auto lval = DataT(lval_i);
        gain += raft::log(lval * invLeft) / raft::log(DataT(2)) * lval * invLen;
      }

      val_i += lval_i;
      auto total_sum = hist[n_bins * c + n_bins - 1].x;
      auto rval_i    = total_sum - lval_i;
      if (rval_i != 0) {
        auto rval = DataT(rval_i);
        gain += raft::log(rval * invRight) / raft::log(DataT(2)) * rval * invLen;
      }

      val_i += rval_i;
      if (val_i != 0) {
        auto val = DataT(val_i) * invLen;
        gain -= val * raft::log(val) / raft::log(DataT(2));
      }
    }

    return gain;
  }

 public:
  HDI DataT
  GainPerSplit(
    BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    switch (criterion) {
      case CRITERION::GINI: return GiniGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::ENTROPY: return EntropyGain(hist, i, n_bins, len, nLeft, nRight);
      default: return -std::numeric_limits<DataT>::max();
    }
  }

  HDI ClassificationObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf, CRITERION criterion)
    : nclasses(nclasses), min_samples_leaf(min_samples_leaf), criterion(criterion)
  {
  }

  DI IdxT NumClasses() const { return nclasses; }

  DI Split<DataT> Gain(
    BinT const* shist, DataT const* squantiles, int col, CountT len, IdxT n_bins) const
  {
    Split<DataT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft  = CountLeft(shist, i, n_bins);
      auto nRight = len - nLeft;
      auto gain   = -std::numeric_limits<DataT>::max();
      if (nLeft >= CountT(min_samples_leaf) && nRight >= CountT(min_samples_leaf)) {
        gain = GainPerSplit(shist, i, n_bins, len, nLeft, nRight);
      }
      sp.update({squantiles[i], col, gain, nLeft});
    }
    return sp;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].x;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].x) / total;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class RegressionObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  using BinT   = AggregateBin;
  using CountT = unsigned long long int;

 private:
  IdxT min_samples_leaf;
  CRITERION criterion;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

  HDI DataT
  MSEGain(BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    auto invLen           = DataT(1.0) / DataT(len);
    auto label_sum        = hist[n_bins - 1].label_sum;
    DataT parent_obj      = -label_sum * label_sum * invLen;
    DataT left_obj        = -(hist[i].label_sum * hist[i].label_sum) / DataT(nLeft);
    DataT right_label_sum = hist[i].label_sum - label_sum;
    DataT right_obj       = -(right_label_sum * right_label_sum) / DataT(nRight);
    DataT gain            = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invLen;

    return gain;
  }

  HDI DataT
  PoissonGain(
    BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    auto invLen          = DataT(1) / DataT(len);
    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -label_sum * raft::log(label_sum * invLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / DataT(nLeft));
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / DataT(nRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  HDI DataT
  GammaGain(BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    auto invLen          = DataT(1) / DataT(len);
    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = DataT(len) * raft::log(label_sum * invLen);
    DataT left_obj   = DataT(nLeft) * raft::log(left_label_sum / DataT(nLeft));
    DataT right_obj  = DataT(nRight) * raft::log(right_label_sum / DataT(nRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  HDI DataT InverseGaussianGain(
    BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -DataT(len) * DataT(len) / label_sum;
    DataT left_obj   = -DataT(nLeft) * DataT(nLeft) / left_label_sum;
    DataT right_obj  = -DataT(nRight) * DataT(nRight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (DataT(2) * DataT(len));

    return gain;
  }

 public:
  HDI DataT
  GainPerSplit(
    BinT const* hist, IdxT i, IdxT n_bins, CountT len, CountT nLeft, CountT nRight) const
  {
    switch (criterion) {
      case CRITERION::MSE: return MSEGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::POISSON: return PoissonGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::GAMMA: return GammaGain(hist, i, n_bins, len, nLeft, nRight);
      case CRITERION::INVERSE_GAUSSIAN:
        return InverseGaussianGain(hist, i, n_bins, len, nLeft, nRight);
      default: return -std::numeric_limits<DataT>::max();
    }
  }

  HDI RegressionObjectiveFunction(IdxT, IdxT min_samples_leaf, CRITERION criterion)
    : min_samples_leaf(min_samples_leaf), criterion(criterion)
  {
  }

  DI IdxT NumClasses() const { return 1; }

  DI Split<DataT> Gain(
    BinT const* shist, DataT const* squantiles, int col, CountT len, IdxT n_bins) const
  {
    Split<DataT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft  = shist[i].count;
      auto nRight = len - nLeft;
      auto gain   = -std::numeric_limits<DataT>::max();
      if (nLeft >= CountT(min_samples_leaf) && nRight >= CountT(min_samples_leaf)) {
        gain = GainPerSplit(shist, i, n_bins, len, nLeft, nRight);
      }
      sp.update({squantiles[i], col, gain, nLeft});
    }
    return sp;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].count;
    }
  }
};
}  // end namespace DT
}  // end namespace ML
