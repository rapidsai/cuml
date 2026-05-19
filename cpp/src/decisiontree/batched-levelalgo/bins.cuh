/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace DT {

struct CountBin {
  int x;
  CountBin(CountBin const&) = default;
  HDI CountBin(int x_) : x(x_) {}
  HDI CountBin() : x(0) {}

  DI static void IncrementHistogram(CountBin* hist, int n_bins, int b, int label)
  {
    auto offset = label * n_bins + b;
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

struct WeightedCountBin {
  double weighted_sum;  // Sum of sample_weight over rows hitting this (bin, class)
  int count;            // Unweighted row count, for min_samples_leaf / _split

  WeightedCountBin(WeightedCountBin const&) = default;
  HDI WeightedCountBin() : weighted_sum(0.0), count(0) {}
  HDI WeightedCountBin(double weighted_sum, int count) : weighted_sum(weighted_sum), count(count) {}

  DI static void IncrementHistogram(
    WeightedCountBin* hist, int n_bins, int b, int label, double weight)
  {
    auto offset = label * n_bins + b;
    WeightedCountBin::AtomicAdd(hist + offset, {weight, 1});
  }
  DI static void AtomicAdd(WeightedCountBin* address, WeightedCountBin val)
  {
    atomicAdd(&address->weighted_sum, val.weighted_sum);
    atomicAdd(&address->count, val.count);
  }
  HDI WeightedCountBin& operator+=(const WeightedCountBin& b)
  {
    weighted_sum += b.weighted_sum;
    count += b.count;
    return *this;
  }
  HDI WeightedCountBin operator+(WeightedCountBin b) const
  {
    b += *this;
    return b;
  }
};
static_assert(sizeof(WeightedCountBin) == 16,
              "WeightedCountBin must be 16 bytes (8 weighted_sum + 4 count + 4 pad)");

struct AggregateBin {
  double label_sum;
  int count;

  AggregateBin(AggregateBin const&) = default;
  HDI AggregateBin() : label_sum(0.0), count(0) {}
  HDI AggregateBin(double label_sum, int count) : label_sum(label_sum), count(count) {}

  DI static void IncrementHistogram(AggregateBin* hist, int n_bins, int b, double label)
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

struct WeightedAggregateBin {
  double label_sum;       // Sum of weight * label
  double weighted_count;  // Sum of weight (sklearn's weighted_n_node_samples)
  int count;              // Unweighted row count

  WeightedAggregateBin(WeightedAggregateBin const&) = default;
  HDI WeightedAggregateBin() : label_sum(0.0), weighted_count(0.0), count(0) {}
  HDI WeightedAggregateBin(double label_sum, double weighted_count, int count)
    : label_sum(label_sum), weighted_count(weighted_count), count(count)
  {
  }

  DI static void IncrementHistogram(
    WeightedAggregateBin* hist, int n_bins, int b, double label, double weight)
  {
    WeightedAggregateBin::AtomicAdd(hist + b, {weight * label, weight, 1});
  }
  DI static void AtomicAdd(WeightedAggregateBin* address, WeightedAggregateBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->weighted_count, val.weighted_count);
    atomicAdd(&address->count, val.count);
  }
  HDI WeightedAggregateBin& operator+=(const WeightedAggregateBin& b)
  {
    label_sum += b.label_sum;
    weighted_count += b.weighted_count;
    count += b.count;
    return *this;
  }
  HDI WeightedAggregateBin operator+(WeightedAggregateBin b) const
  {
    b += *this;
    return b;
  }
};
static_assert(
  sizeof(WeightedAggregateBin) == 24,
  "WeightedAggregateBin must be 24 bytes (8 label_sum + 8 weighted_count + 4 count + 4 pad)");
}  // namespace DT
}  // namespace ML
