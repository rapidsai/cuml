/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace DT {

struct CountBin {
  // double covers both the unweighted count path (weight=1.0) and the
  // per-sample-weighted path; 32-bit int would overflow on large weighted counts.
  double x;
  CountBin(CountBin const&) = default;
  HDI CountBin(double x_) : x(x_) {}
  HDI CountBin() : x(0.0) {}

  DI static void IncrementHistogram(CountBin* hist, int n_bins, int b, int label, double weight)
  {
    auto offset = label * n_bins + b;
    CountBin::AtomicAdd(hist + offset, {weight});
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
  // `label_sum` carries `label * weight`; `count` stays unweighted so
  // `min_samples_leaf` checks operate on the integer sample count.
  double label_sum;
  int count;

  AggregateBin(AggregateBin const&) = default;
  HDI AggregateBin() : label_sum(0.0), count(0) {}
  HDI AggregateBin(double label_sum, int count) : label_sum(label_sum), count(count) {}

  DI static void IncrementHistogram(
    AggregateBin* hist, int n_bins, int b, double label, double weight)
  {
    AggregateBin::AtomicAdd(hist + b, {label * weight, 1});
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
static_assert(sizeof(AggregateBin) == 16, "AggregateBin layout drift");
}  // namespace DT
}  // namespace ML
