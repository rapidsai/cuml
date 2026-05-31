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
  HDI DataT
  GainPerSplit(BinT* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft, double W_total, double W_left)
  {
    IdxT nRight = len - nLeft;
    // min_samples_leaf gate stays on the unweighted counts: sklearn enforces
    // min_samples_leaf against the integer left/right sample counts.
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    // Skip splits where one side has all-zero weights but nonzero sample count.
    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    // Weighted Gini proxy: argmax of sum_c w_left_c^2/W_left + w_right_c^2/W_right.
    // The leading 1/W_total scaling drops a constant term that's identical
    // across split candidates within a node.
    auto invW      = DataT(1.0) / DataT(W_total);
    auto invWLeft  = DataT(1.0) / DataT(W_left);
    auto invWRight = DataT(1.0) / DataT(W_right);
    auto gain      = DataT(0.0);

    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval      = DataT(hist[n_bins * j + i].x);
      auto total_sum = DataT(hist[n_bins * j + n_bins - 1].x);
      auto rval      = total_sum - lval;

      gain += lval * invWLeft * lval * invW;
      gain += rval * invWRight * rval * invW;

      auto val = total_sum * invW;
      gain -= val * val;
    }

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT* shist, DataT* squantiles, IdxT col, IdxT len, IdxT n_bins, int const* unweighted_cdf)
  {
    // Hoist W_total out of the per-bin loop. The total weighted node sample
    // count is the last-bin CDF value summed over classes.
    double W_total = 0.0;
    for (IdxT j = 0; j < nclasses; ++j)
      W_total += shist[n_bins * j + n_bins - 1].x;

    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      // Read the unweighted prefix-count directly: under fractional
      // sample_weight the per-class weighted CDF can't carry the integer
      // sample count that min_samples_leaf and Split::nLeft need.
      IdxT nLeft = unweighted_cdf[i];

      // Weighted left-side total at bin i, summed over classes.
      double W_left = 0.0;
      for (IdxT j = 0; j < nclasses; ++j)
        W_left += shist[n_bins * j + i].x;

      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft, W_total, W_left), nLeft});
    }
    return sp;
  }
  static DI void SetLeafVector(BinT const* shist,
                               int nclasses,
                               DataT* out,
                               double /* weighted_total, unused for classifier */)
  {
    // Output probability
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].x;
    }
    if (total <= 0.0) {
      // All zero-weight leaf: emit a uniform distribution so downstream consumers
      // see a finite vector and so argmax stays deterministic across builds.
      DataT uniform = DataT(1.0) / DataT(nclasses);
      for (int i = 0; i < nclasses; i++)
        out[i] = uniform;
      return;
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
  HDI DataT GainPerSplit(
    BinT const* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft, double W_total, double W_left)
  {
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    // Skip splits where one side has all-zero weights but nonzero sample count.
    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    auto invWLeft  = DataT(1.0) / DataT(W_left);
    auto invWRight = DataT(1.0) / DataT(W_right);
    auto invW      = DataT(1.0) / DataT(W_total);
    auto gain      = DataT(0.0);

    for (IdxT c = 0; c < nclasses; ++c) {
      auto lval_i = hist[n_bins * c + i].x;
      if (lval_i != 0) {
        auto lval = DataT(lval_i);
        gain += raft::log(lval * invWLeft) / raft::log(DataT(2)) * lval * invW;
      }

      auto total_sum = hist[n_bins * c + n_bins - 1].x;
      auto rval_i    = total_sum - lval_i;
      if (rval_i != 0) {
        auto rval = DataT(rval_i);
        gain += raft::log(rval * invWRight) / raft::log(DataT(2)) * rval * invW;
      }

      if (total_sum != 0) {
        auto val = DataT(total_sum) * invW;
        gain -= val * raft::log(val) / raft::log(DataT(2));
      }
    }

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT* scdf_labels,
                             DataT* squantiles,
                             IdxT col,
                             IdxT len,
                             IdxT n_bins,
                             int const* unweighted_cdf)
  {
    double W_total = 0.0;
    for (IdxT c = 0; c < nclasses; ++c)
      W_total += scdf_labels[n_bins * c + n_bins - 1].x;

    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      IdxT nLeft = unweighted_cdf[i];

      double W_left = 0.0;
      for (IdxT c = 0; c < nclasses; ++c)
        W_left += scdf_labels[n_bins * c + i].x;

      sp.update({squantiles[i],
                 col,
                 GainPerSplit(scdf_labels, i, n_bins, len, nLeft, W_total, W_left),
                 nLeft});
    }
    return sp;
  }
  static DI void SetLeafVector(BinT const* shist,
                               int nclasses,
                               DataT* out,
                               double /* weighted_total, unused for classifier */)
  {
    // Output probability
    double total = 0.0;
    for (int i = 0; i < nclasses; i++) {
      total += shist[i].x;
    }
    if (total <= 0.0) {
      DataT uniform = DataT(1.0) / DataT(nclasses);
      for (int i = 0; i < nclasses; i++)
        out[i] = uniform;
      return;
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
  HDI DataT GainPerSplit(BinT const* hist,
                         IdxT i,
                         IdxT n_bins,
                         IdxT len,
                         IdxT nLeft,
                         double W_total,
                         double W_left) const
  {
    IdxT nRight{len - nLeft};
    // min_samples_leaf gate stays on the unweighted integer count: sklearn
    // enforces min_samples_leaf against the sample count, not the weight sum.
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
      return -std::numeric_limits<DataT>::max();
    }
    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    // Weighted MSE proxy (sklearn 1.7.2 _criterion.pyx:1089-1118): replace n
    // with sum(weight) in every denominator so leaf-mean is
    // sum(label*weight)/sum(weight).
    auto invW             = DataT(1.0) / DataT(W_total);
    auto label_sum        = hist[n_bins - 1].label_sum;
    DataT parent_obj      = -DataT(label_sum) * DataT(label_sum) * invW;
    DataT left_obj        = -DataT(hist[i].label_sum * hist[i].label_sum) / DataT(W_left);
    DataT right_label_sum = label_sum - hist[i].label_sum;
    DataT right_obj       = -(right_label_sum * right_label_sum) / DataT(W_right);
    DataT gain            = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invW;
    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist,
                             DataT const* squantiles,
                             IdxT col,
                             IdxT len,
                             IdxT n_bins,
                             double const* weighted_cdf) const
  {
    // AggregateBin.count is the unweighted sample count (see bins.cuh); the
    // weighted CDF lives in the parallel companion buffer.
    double W_total = weighted_cdf[n_bins - 1];
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft    = shist[i].count;
      double W_left = weighted_cdf[i];
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft, W_total, W_left), nLeft});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out, double weighted_total)
  {
    if (weighted_total <= 0.0) {
      for (int i = 0; i < nclasses; i++)
        out[i] = DataT(0);
      return;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].label_sum / weighted_total);
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
  HDI DataT GainPerSplit(BinT const* hist,
                         IdxT i,
                         IdxT n_bins,
                         IdxT len,
                         IdxT nLeft,
                         double W_total,
                         double W_left) const
  {
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // Weighted Poisson half-deviance (sklearn 1.7.2 _criterion.pyx:1597-1642):
    // swap sample-count denominators for sum(weight) so the predicted-mean per
    // leaf is sum(label*weight)/sum(weight).
    auto invW        = DataT(1.0) / DataT(W_total);
    DataT parent_obj = -label_sum * raft::log(label_sum * invW);
    DataT left_obj   = -left_label_sum * raft::log(left_label_sum / DataT(W_left));
    DataT right_obj  = -right_label_sum * raft::log(right_label_sum / DataT(W_right));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invW;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist,
                             DataT const* squantiles,
                             IdxT col,
                             IdxT len,
                             IdxT n_bins,
                             double const* weighted_cdf) const
  {
    double W_total = weighted_cdf[n_bins - 1];
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft    = shist[i].count;
      double W_left = weighted_cdf[i];
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft, W_total, W_left), nLeft});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out, double weighted_total)
  {
    if (weighted_total <= 0.0) {
      for (int i = 0; i < nclasses; i++)
        out[i] = DataT(0);
      return;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].label_sum / weighted_total);
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
  HDI DataT GainPerSplit(BinT const* hist,
                         IdxT i,
                         IdxT n_bins,
                         IdxT len,
                         IdxT nLeft,
                         double W_total,
                         double W_left) const
  {
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    DataT label_sum       = hist[n_bins - 1].label_sum;
    DataT left_label_sum  = (hist[i].label_sum);
    DataT right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // Weighted Gamma half-deviance: the n coefficients become sum(weight) so
    // the formula stays scale-equivariant under sample reweighting.
    auto invW        = DataT(1.0) / DataT(W_total);
    DataT parent_obj = DataT(W_total) * raft::log(label_sum * invW);
    DataT left_obj   = DataT(W_left) * raft::log(left_label_sum / DataT(W_left));
    DataT right_obj  = DataT(W_right) * raft::log(right_label_sum / DataT(W_right));
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invW;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist,
                             DataT const* squantiles,
                             IdxT col,
                             IdxT len,
                             IdxT n_bins,
                             double const* weighted_cdf) const
  {
    double W_total = weighted_cdf[n_bins - 1];
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft    = shist[i].count;
      double W_left = weighted_cdf[i];
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft, W_total, W_left), nLeft});
    }
    return sp;
  }
  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out, double weighted_total)
  {
    if (weighted_total <= 0.0) {
      for (int i = 0; i < nclasses; i++)
        out[i] = DataT(0);
      return;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].label_sum / weighted_total);
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
  HDI DataT GainPerSplit(BinT const* hist,
                         IdxT i,
                         IdxT n_bins,
                         IdxT len,
                         IdxT nLeft,
                         double W_total,
                         double W_left) const
  {
    IdxT nRight = len - nLeft;
    if (nLeft < min_samples_leaf || nRight < min_samples_leaf)
      return -std::numeric_limits<DataT>::max();

    double W_right = W_total - W_left;
    if (W_left <= 0.0 || W_right <= 0.0) return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // Weighted Inverse Gaussian half-deviance: substitute n with sum(weight)
    // so the formula stays scale-equivariant under sample reweighting.
    DataT parent_obj = -DataT(W_total) * DataT(W_total) / label_sum;
    DataT left_obj   = -DataT(W_left) * DataT(W_left) / left_label_sum;
    DataT right_obj  = -DataT(W_right) * DataT(W_right) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (DataT(2.0) * DataT(W_total));

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT const* shist,
                             DataT const* squantiles,
                             IdxT col,
                             IdxT len,
                             IdxT n_bins,
                             double const* weighted_cdf) const
  {
    double W_total = weighted_cdf[n_bins - 1];
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeft    = shist[i].count;
      double W_left = weighted_cdf[i];
      sp.update(
        {squantiles[i], col, GainPerSplit(shist, i, n_bins, len, nLeft, W_total, W_left), nLeft});
    }
    return sp;
  }
  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out, double weighted_total)
  {
    if (weighted_total <= 0.0) {
      for (int i = 0; i < nclasses; i++)
        out[i] = DataT(0);
      return;
    }
    for (int i = 0; i < nclasses; i++) {
      out[i] = DataT(shist[i].label_sum / weighted_total);
    }
  }
};
}  // end namespace DT
}  // end namespace ML
