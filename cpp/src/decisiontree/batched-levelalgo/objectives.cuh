/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "dataset.h"
#include "split.cuh"

#include <cub/cub.cuh>

#include <limits>

namespace ML {
namespace DT {

template <typename DataT_, typename LabelT_, typename IdxT_>
class GiniObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf;

 public:
  using BinT = CountBin;
  GiniObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : nclasses(nclasses), min_samples_leaf(min_samples_leaf)
  {
  }

  DI IdxT NumClasses() const { return nclasses; }

  /**
   * @brief compute the gini impurity reduction for each split
   */
  HDI DataT GainPerSplit(BinT* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft)
  {
    IdxT nRight         = len - nLeft;
    constexpr DataT One = DataT(1.0);
    auto invLen         = One / len;
    auto invLeft        = One / nLeft;
    auto invRight       = One / nRight;
    auto gain           = DataT(0.0);

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    for (IdxT j = 0; j < nclasses; ++j) {
      int val_i   = 0;
      auto lval_i = hist[n_bins * j + i].x;
      auto lval   = DataT(lval_i);
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

  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* squantiles, IdxT col, IdxT len, IdxT n_bins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      IdxT nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += shist[n_bins * j + i].x;
      }
      sp.update({squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  // left[j].x and parent[j].x are per-class unweighted counts; right side is
  // computed inline as parent[j].x - left[j].x.
  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft = 0;
    IdxT len   = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += left[j].x;
      len += parent[j].x;
    }
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    constexpr DataT One = DataT(1.0);
    auto invLen         = One / len;
    auto invLeft        = One / nLeft;
    auto invRight       = One / nRight;
    auto gain           = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval_i = left[j].x;
      auto lval   = DataT(lval_i);
      gain += lval * invLeft * lval * invLen;
      auto rval_i = parent[j].x - lval_i;
      auto rval   = DataT(rval_i);
      gain += rval * invRight * rval * invLen;
      auto val = DataT(parent[j].x) * invLen;
      gain -= val * val;
    }
    return gain;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    int total = 0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].x;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].x) / total;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class EntropyObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf;

 public:
  using BinT = CountBin;
  EntropyObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : nclasses(nclasses), min_samples_leaf(min_samples_leaf)
  {
  }
  DI IdxT NumClasses() const { return nclasses; }

  /**
   * @brief compute the Entropy (or information gain) for each split
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft)
  {
    IdxT nRight{len - nLeft};
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
        auto lval_i = hist[n_bins * c + i].x;
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
  }

  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* squantiles, IdxT col, IdxT len, IdxT n_bins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      IdxT nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[n_bins * j + i].x;
      }
      sp.update({squantiles[i], col, GainPerSplit(scdf_labels, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft = 0;
    IdxT len   = 0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeft += left[j].x;
      len += parent[j].x;
    }
    IdxT nRight = len - nLeft;
    auto gain   = DataT(0.0);
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto invLeft  = DataT(1.0) / nLeft;
    auto invRight = DataT(1.0) / nRight;
    auto invLen   = DataT(1.0) / len;
    for (IdxT c = 0; c < nclasses; ++c) {
      auto lval_i = left[c].x;
      if (lval_i != 0) {
        auto lval = DataT(lval_i);
        gain += raft::log(lval * invLeft) / raft::log(DataT(2)) * lval * invLen;
      }
      auto rval_i = parent[c].x - lval_i;
      if (rval_i != 0) {
        auto rval = DataT(rval_i);
        gain += raft::log(rval * invRight) / raft::log(DataT(2)) * rval * invLen;
      }
      auto val_i = parent[c].x;
      if (val_i != 0) {
        auto val = DataT(val_i) * invLen;
        gain -= val * raft::log(val) / raft::log(DataT(2));
      }
    }
    return gain;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    int total = 0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].x;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].x) / total;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class MSEObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  using BinT   = AggregateBin;

 private:
  IdxT min_samples_leaf;

 public:
  HDI MSEObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf(min_samples_leaf)
  {
  }

  /**
   * @brief compute the Mean squared error impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy mean squared error reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       mean-squared errors.
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft) const
  {
    auto gain{DataT(0)};
    IdxT nRight{len - nLeft};
    auto invLen = DataT(1.0) / len;
    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      auto label_sum        = hist[n_bins - 1].label_sum;
      DataT parent_obj      = -label_sum * label_sum * invLen;
      DataT left_obj        = -(hist[i].label_sum * hist[i].label_sum) / nLeft;
      DataT right_label_sum = hist[i].label_sum - label_sum;
      DataT right_obj       = -(right_label_sum * right_label_sum) / nRight;
      gain                  = parent_obj - (left_obj + right_obj);
      gain *= DataT(0.5) * invLen;

      return gain;
    }
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  // Regressor side stats are single AggregateBin tuples; only left[0] and
  // parent[0] are consulted (per_side_cells = NumClasses() = 1).
  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft  = left[0].count;
    IdxT len    = parent[0].count;
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto invLen          = DataT(1.0) / len;
    auto label_sum       = parent[0].label_sum;
    auto left_label_sum  = left[0].label_sum;
    auto right_label_sum = label_sum - left_label_sum;
    DataT parent_obj     = -label_sum * label_sum * invLen;
    DataT left_obj       = -(left_label_sum * left_label_sum) / nLeft;
    DataT right_obj      = -(right_label_sum * right_label_sum) / nRight;
    DataT gain           = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class PoissonObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  using BinT   = AggregateBin;

 private:
  IdxT min_samples_leaf;

 public:
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

  HDI PoissonObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf(min_samples_leaf)
  {
  }

  /**
   * @brief compute the poisson impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy poisson half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       poisson half deviances.
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft) const
  {
    // get the lens'
    IdxT nRight = len - nLeft;
    auto invLen = DataT(1) / len;

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = -label_sum * raft::log(label_sum * invLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / nLeft);
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / nRight);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft  = left[0].count;
    IdxT len    = parent[0].count;
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto invLen          = DataT(1) / len;
    auto label_sum       = parent[0].label_sum;
    auto left_label_sum  = left[0].label_sum;
    auto right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = -label_sum * raft::log(label_sum * invLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / nLeft);
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / nRight);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class GammaObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  using BinT                 = AggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

 private:
  IdxT min_samples_leaf;

 public:
  HDI GammaObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf{min_samples_leaf}
  {
  }

  /**
   * @brief compute the gamma impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy gamma half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       gamma half deviances.
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft) const
  {
    IdxT nRight = len - nLeft;
    auto invLen = DataT(1) / len;

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    DataT label_sum       = hist[n_bins - 1].label_sum;
    DataT left_label_sum  = (hist[i].label_sum);
    DataT right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = len * raft::log(label_sum * invLen);
    DataT left_obj   = nLeft * raft::log(left_label_sum / nLeft);
    DataT right_obj  = nRight * raft::log(right_label_sum / nRight);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft  = left[0].count;
    IdxT len    = parent[0].count;
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto invLen           = DataT(1) / len;
    DataT label_sum       = parent[0].label_sum;
    DataT left_label_sum  = left[0].label_sum;
    DataT right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = len * raft::log(label_sum * invLen);
    DataT left_obj   = nLeft * raft::log(left_label_sum / nLeft);
    DataT right_obj  = nRight * raft::log(right_label_sum / nRight);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class InverseGaussianObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  using BinT                 = AggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

 private:
  IdxT min_samples_leaf;

 public:
  HDI InverseGaussianObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf{min_samples_leaf}
  {
  }

  /**
   * @brief compute the inverse gaussian impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy inverse gaussian half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       inverse gaussian deviances.
   */
  HDI DataT GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft) const
  {
    // get the lens'
    IdxT nRight = len - nLeft;

    // if there aren't enough samples in this split, don't bother!
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = -DataT(len) * DataT(len) / label_sum;
    DataT left_obj   = -DataT(nLeft) * DataT(nLeft) / left_label_sum;
    DataT right_obj  = -DataT(nRight) * DataT(nRight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (2 * len);

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      sp.update({squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft), nLeft});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeft  = left[0].count;
    IdxT len    = parent[0].count;
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto label_sum       = parent[0].label_sum;
    auto left_label_sum  = left[0].label_sum;
    auto right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = -DataT(len) * DataT(len) / label_sum;
    DataT left_obj   = -DataT(nLeft) * DataT(nLeft) / left_label_sum;
    DataT right_obj  = -DataT(nRight) * DataT(nRight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (2 * len);
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedGiniObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf;

 public:
  using BinT = WeightedCountBin;
  WeightedGiniObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : nclasses(nclasses), min_samples_leaf(min_samples_leaf)
  {
  }

  DI IdxT NumClasses() const { return nclasses; }

  // wLeft is the weighted left sum (from Gain); nLeftCount the unweighted
  // left count (drives min_samples_leaf); weighted parent total from last bin.
  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    double wLen = 0.0;
    for (IdxT j = 0; j < nclasses; ++j)
      wLen += hist[n_bins * j + n_bins - 1].weighted_sum;
    double wRight = wLen - wLeft;
    // A split whose unweighted nLeftCount passes min_samples_leaf can still
    // have an all-zero-weight side; guard the weighted divisors.
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();

    auto invWLen   = DataT(1.0) / DataT(wLen);
    auto invWLeft  = DataT(1.0) / DataT(wLeft);
    auto invWRight = DataT(1.0) / DataT(wRight);
    auto gain      = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval = DataT(hist[n_bins * j + i].weighted_sum);
      gain += lval * invWLeft * lval * invWLen;

      auto total_sum = DataT(hist[n_bins * j + n_bins - 1].weighted_sum);
      auto rval      = total_sum - lval;
      gain += rval * invWRight * rval * invWLen;

      auto val = (lval + rval) * invWLen;
      gain -= val * val;
    }
    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* squantiles, IdxT col, IdxT len, IdxT n_bins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = 0.0;
      IdxT nLeftCount = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        wLeft += shist[n_bins * j + i].weighted_sum;
        nLeftCount += shist[n_bins * j + i].count;
      }
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeftCount, wLeft), nLeftCount});
    }
    return sp;
  }

  // ExtraTrees counterpart of GainPerSplit. Reads whole-side stats from
  // left[j].weighted_sum / .count and parent[j].weighted_sum / .count.
  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount = 0;
    IdxT len        = 0;
    double wLeft    = 0.0;
    double wLen     = 0.0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeftCount += left[j].count;
      len += parent[j].count;
      wLeft += left[j].weighted_sum;
      wLen += parent[j].weighted_sum;
    }
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    double wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();

    auto invWLen   = DataT(1.0) / DataT(wLen);
    auto invWLeft  = DataT(1.0) / DataT(wLeft);
    auto invWRight = DataT(1.0) / DataT(wRight);
    auto gain      = DataT(0.0);
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval = DataT(left[j].weighted_sum);
      gain += lval * invWLeft * lval * invWLen;
      auto rval = DataT(parent[j].weighted_sum - left[j].weighted_sum);
      gain += rval * invWRight * rval * invWLen;
      auto val = DataT(parent[j].weighted_sum) * invWLen;
      gain -= val * val;
    }
    return gain;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].weighted_sum;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].weighted_sum) / DataT(total);
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedEntropyObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf;

 public:
  using BinT = WeightedCountBin;
  WeightedEntropyObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : nclasses(nclasses), min_samples_leaf(min_samples_leaf)
  {
  }
  DI IdxT NumClasses() const { return nclasses; }

  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    auto gain{DataT(0.0)};
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    } else {
      double wLen = 0.0;
      for (IdxT c = 0; c < nclasses; ++c)
        wLen += hist[n_bins * c + n_bins - 1].weighted_sum;
      double wRight = wLen - wLeft;
      if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();

      auto invWLeft  = DataT(1.0) / DataT(wLeft);
      auto invWRight = DataT(1.0) / DataT(wRight);
      auto invWLen   = DataT(1.0) / DataT(wLen);
      for (IdxT c = 0; c < nclasses; ++c) {
        auto lval = DataT(hist[n_bins * c + i].weighted_sum);
        if (lval != DataT(0))
          gain += raft::log(lval * invWLeft) / raft::log(DataT(2)) * lval * invWLen;

        auto total_sum = DataT(hist[n_bins * c + n_bins - 1].weighted_sum);
        auto rval      = total_sum - lval;
        if (rval != DataT(0))
          gain += raft::log(rval * invWRight) / raft::log(DataT(2)) * rval * invWLen;

        auto val = (lval + rval) * invWLen;
        if (val != DataT(0)) gain -= val * raft::log(val) / raft::log(DataT(2));
      }
      return gain;
    }
  }

  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* squantiles, IdxT col, IdxT len, IdxT n_bins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = 0.0;
      IdxT nLeftCount = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        wLeft += scdf_labels[n_bins * j + i].weighted_sum;
        nLeftCount += scdf_labels[n_bins * j + i].count;
      }
      sp.update({squantiles[i],
                 col,
                 GainPerSplit(scdf_labels, i, n_bins, len, nLeftCount, wLeft),
                 nLeftCount});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount = 0;
    IdxT len        = 0;
    double wLeft    = 0.0;
    double wLen     = 0.0;
    for (IdxT j = 0; j < nclasses; ++j) {
      nLeftCount += left[j].count;
      len += parent[j].count;
      wLeft += left[j].weighted_sum;
      wLen += parent[j].weighted_sum;
    }
    IdxT nRightCount = len - nLeftCount;
    auto gain        = DataT(0.0);
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    double wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();

    auto invWLeft  = DataT(1.0) / DataT(wLeft);
    auto invWRight = DataT(1.0) / DataT(wRight);
    auto invWLen   = DataT(1.0) / DataT(wLen);
    for (IdxT c = 0; c < nclasses; ++c) {
      auto lval = DataT(left[c].weighted_sum);
      if (lval != DataT(0))
        gain += raft::log(lval * invWLeft) / raft::log(DataT(2)) * lval * invWLen;
      auto rval = DataT(parent[c].weighted_sum - left[c].weighted_sum);
      if (rval != DataT(0))
        gain += raft::log(rval * invWRight) / raft::log(DataT(2)) * rval * invWLen;
      auto val = DataT(parent[c].weighted_sum) * invWLen;
      if (val != DataT(0)) gain -= val * raft::log(val) / raft::log(DataT(2));
    }
    return gain;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].weighted_sum;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].weighted_sum) / DataT(total);
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedMSEObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  using BinT   = WeightedAggregateBin;

 private:
  IdxT min_samples_leaf;

 public:
  HDI WeightedMSEObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf(min_samples_leaf)
  {
  }

  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto wLen   = hist[n_bins - 1].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen = DataT(1.0) / DataT(wLen);

    auto label_sum      = hist[n_bins - 1].label_sum;
    DataT parent_obj    = -label_sum * label_sum * invWLen;
    DataT left_obj      = -(hist[i].label_sum * hist[i].label_sum) / DataT(wLeft);
    DataT right_lbl_sum = label_sum - hist[i].label_sum;
    DataT right_obj     = -(right_lbl_sum * right_lbl_sum) / DataT(wRight);
    DataT gain          = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invWLen;
    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = shist[i].weighted_count;
      IdxT nLeftCount = shist[i].count;
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeftCount, wLeft), nLeftCount});
    }
    return sp;
  }

  // Regressor side stats are single WeightedAggregateBin tuples; only left[0]
  // and parent[0] are consulted (per_side_cells = NumClasses() = 1).
  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount  = left[0].count;
    IdxT len         = parent[0].count;
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto wLeft  = left[0].weighted_count;
    auto wLen   = parent[0].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen        = DataT(1.0) / DataT(wLen);
    auto label_sum      = parent[0].label_sum;
    DataT parent_obj    = -label_sum * label_sum * invWLen;
    DataT left_obj      = -(left[0].label_sum * left[0].label_sum) / DataT(wLeft);
    DataT right_lbl_sum = label_sum - left[0].label_sum;
    DataT right_obj     = -(right_lbl_sum * right_lbl_sum) / DataT(wRight);
    DataT gain          = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invWLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].weighted_count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedPoissonObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  using BinT                 = WeightedAggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

 private:
  IdxT min_samples_leaf;

 public:
  HDI WeightedPoissonObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf(min_samples_leaf)
  {
  }

  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto wLen   = hist[n_bins - 1].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen = DataT(1) / DataT(wLen);

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = hist[i].label_sum;
    auto right_label_sum = label_sum - hist[i].label_sum;

    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -label_sum * raft::log(label_sum * invWLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / DataT(wLeft));
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / DataT(wRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invWLen;
    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = shist[i].weighted_count;
      IdxT nLeftCount = shist[i].count;
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeftCount, wLeft), nLeftCount});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount  = left[0].count;
    IdxT len         = parent[0].count;
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto wLeft  = left[0].weighted_count;
    auto wLen   = parent[0].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen         = DataT(1) / DataT(wLen);
    auto label_sum       = parent[0].label_sum;
    auto left_label_sum  = left[0].label_sum;
    auto right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = -label_sum * raft::log(label_sum * invWLen);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / DataT(wLeft));
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / DataT(wRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invWLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].weighted_count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedGammaObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  using BinT                 = WeightedAggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

 private:
  IdxT min_samples_leaf;

 public:
  HDI WeightedGammaObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf{min_samples_leaf}
  {
  }

  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto wLen   = hist[n_bins - 1].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen = DataT(1) / DataT(wLen);

    DataT label_sum       = hist[n_bins - 1].label_sum;
    DataT left_label_sum  = hist[i].label_sum;
    DataT right_label_sum = label_sum - hist[i].label_sum;

    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = DataT(wLen) * raft::log(label_sum * invWLen);
    DataT left_obj   = DataT(wLeft) * raft::log(left_label_sum / DataT(wLeft));
    DataT right_obj  = DataT(wRight) * raft::log(right_label_sum / DataT(wRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invWLen;
    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = shist[i].weighted_count;
      IdxT nLeftCount = shist[i].count;
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeftCount, wLeft), nLeftCount});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount  = left[0].count;
    IdxT len         = parent[0].count;
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto wLeft  = left[0].weighted_count;
    auto wLen   = parent[0].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto invWLen          = DataT(1) / DataT(wLen);
    DataT label_sum       = parent[0].label_sum;
    DataT left_label_sum  = left[0].label_sum;
    DataT right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = DataT(wLen) * raft::log(label_sum * invWLen);
    DataT left_obj   = DataT(wLeft) * raft::log(left_label_sum / DataT(wLeft));
    DataT right_obj  = DataT(wRight) * raft::log(right_label_sum / DataT(wRight));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invWLen;
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].weighted_count;
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class WeightedInverseGaussianObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  using BinT                 = WeightedAggregateBin;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

 private:
  IdxT min_samples_leaf;

 public:
  HDI WeightedInverseGaussianObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf)
    : min_samples_leaf{min_samples_leaf}
  {
  }

  HDI DataT
  GainPerSplit(BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeftCount, double wLeft) const
  {
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    auto wLen   = hist[n_bins - 1].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = hist[i].label_sum;
    auto right_label_sum = label_sum - hist[i].label_sum;

    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    DataT parent_obj = -DataT(wLen) * DataT(wLen) / label_sum;
    DataT left_obj   = -DataT(wLeft) * DataT(wLeft) / left_label_sum;
    DataT right_obj  = -DataT(wRight) * DataT(wRight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (DataT(2) * DataT(wLen));
    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      double wLeft    = shist[i].weighted_count;
      IdxT nLeftCount = shist[i].count;
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeftCount, wLeft), nLeftCount});
    }
    return sp;
  }

  HDI DataT GainFromSideStats(BinT const* left, BinT const* parent) const
  {
    IdxT nLeftCount  = left[0].count;
    IdxT len         = parent[0].count;
    IdxT nRightCount = len - nLeftCount;
    if (nLeftCount < min_samples_leaf || nRightCount < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();
    auto wLeft  = left[0].weighted_count;
    auto wLen   = parent[0].weighted_count;
    auto wRight = wLen - wLeft;
    if (wLeft <= 0.0 || wRight <= 0.0) return -std::numeric_limits<DataT>::max();
    auto label_sum       = parent[0].label_sum;
    auto left_label_sum  = left[0].label_sum;
    auto right_label_sum = label_sum - left_label_sum;
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();
    DataT parent_obj = -DataT(wLen) * DataT(wLen) / label_sum;
    DataT left_obj   = -DataT(wLeft) * DataT(wLeft) / left_label_sum;
    DataT right_obj  = -DataT(wRight) * DataT(wRight) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (DataT(2) * DataT(wLen));
    return gain;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      out[i] = shist[i].label_sum / shist[i].weighted_count;
    }
  }
};
}  // end namespace DT
}  // end namespace ML
