/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cuda/std/array>

namespace ML {
namespace DT {

using BinCountT = unsigned long long int;
static_assert(sizeof(BinCountT) == 8, "BinCountT must be 64 bits");

struct ClassificationBin {
  BinCountT count;

  ClassificationBin(ClassificationBin const&) = default;
  HDI ClassificationBin(BinCountT count_) : count(count_) {}
  HDI ClassificationBin() : count(0) {}

  DI static void IncrementHistogram(ClassificationBin* hist, int n_bins, int b, int label, double)
  {
    auto offset = label * n_bins + b;
    ClassificationBin::AtomicAdd(hist + offset, {1});
  }
  DI static void AtomicAdd(ClassificationBin* address, ClassificationBin val)
  {
    atomicAdd(&address->count, val.count);
  }
  HDI BinCountT Count() const { return count; }
  HDI double Weight() const { return static_cast<double>(count); }
  HDI auto ToReductionBuffer() const
  {
    return cuda::std::array<double, 1>{static_cast<double>(count)};
  }
  HDI static ClassificationBin FromReductionBuffer(cuda::std::array<double, 1> const& buffer)
  {
    return {static_cast<BinCountT>(buffer[0])};
  }
  HDI ClassificationBin& operator+=(const ClassificationBin& b)
  {
    count += b.count;
    return *this;
  }
  HDI ClassificationBin operator+(ClassificationBin b) const
  {
    b += *this;
    return b;
  }
};

struct WeightedClassificationBin {
  BinCountT count;
  double weight;

  WeightedClassificationBin(WeightedClassificationBin const&) = default;
  HDI WeightedClassificationBin(BinCountT count_, double weight_) : count(count_), weight(weight_)
  {
  }
  HDI WeightedClassificationBin() : count(0), weight(0.0) {}

  DI static void IncrementHistogram(
    WeightedClassificationBin* hist, int n_bins, int b, int label, double weight)
  {
    auto offset = label * n_bins + b;
    WeightedClassificationBin::AtomicAdd(hist + offset, {1, weight});
  }
  DI static void AtomicAdd(WeightedClassificationBin* address, WeightedClassificationBin val)
  {
    atomicAdd(&address->count, val.count);
    atomicAdd(&address->weight, val.weight);
  }
  HDI BinCountT Count() const { return count; }
  HDI double Weight() const { return weight; }
  HDI auto ToReductionBuffer() const
  {
    return cuda::std::array<double, 2>{static_cast<double>(count), weight};
  }
  HDI static WeightedClassificationBin FromReductionBuffer(
    cuda::std::array<double, 2> const& buffer)
  {
    return {static_cast<BinCountT>(buffer[0]), buffer[1]};
  }
  HDI WeightedClassificationBin& operator+=(const WeightedClassificationBin& b)
  {
    count += b.count;
    weight += b.weight;
    return *this;
  }
  HDI WeightedClassificationBin operator+(WeightedClassificationBin b) const
  {
    b += *this;
    return b;
  }
};

struct RegressionBin {
  double label_sum;
  BinCountT count;

  RegressionBin(RegressionBin const&) = default;
  HDI RegressionBin() : label_sum(0.0), count(0) {}
  HDI RegressionBin(double label_sum, BinCountT count) : label_sum(label_sum), count(count) {}

  DI static void IncrementHistogram(RegressionBin* hist, int n_bins, int b, double label, double)
  {
    RegressionBin::AtomicAdd(hist + b, {label, 1});
  }
  DI static void AtomicAdd(RegressionBin* address, RegressionBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->count, val.count);
  }
  HDI double LabelSum() const { return label_sum; }
  HDI BinCountT Count() const { return count; }
  HDI double Weight() const { return static_cast<double>(count); }
  HDI auto ToReductionBuffer() const
  {
    return cuda::std::array<double, 2>{label_sum, static_cast<double>(count)};
  }
  HDI static RegressionBin FromReductionBuffer(cuda::std::array<double, 2> const& buffer)
  {
    return {buffer[0], static_cast<BinCountT>(buffer[1])};
  }
  HDI RegressionBin& operator+=(const RegressionBin& b)
  {
    label_sum += b.label_sum;
    count += b.count;
    return *this;
  }
  HDI RegressionBin operator+(RegressionBin b) const
  {
    b += *this;
    return b;
  }
};

struct WeightedRegressionBin {
  double label_sum;
  BinCountT count;
  double weight;

  WeightedRegressionBin(WeightedRegressionBin const&) = default;
  HDI WeightedRegressionBin() : label_sum(0.0), count(0), weight(0.0) {}
  HDI WeightedRegressionBin(double label_sum, BinCountT count, double weight)
    : label_sum(label_sum), count(count), weight(weight)
  {
  }

  DI static void IncrementHistogram(
    WeightedRegressionBin* hist, int n_bins, int b, double label, double weight)
  {
    WeightedRegressionBin::AtomicAdd(hist + b, {label * weight, 1, weight});
  }
  DI static void AtomicAdd(WeightedRegressionBin* address, WeightedRegressionBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->count, val.count);
    atomicAdd(&address->weight, val.weight);
  }
  HDI double LabelSum() const { return label_sum; }
  HDI BinCountT Count() const { return count; }
  HDI double Weight() const { return weight; }
  HDI auto ToReductionBuffer() const
  {
    return cuda::std::array<double, 3>{label_sum, static_cast<double>(count), weight};
  }
  HDI static WeightedRegressionBin FromReductionBuffer(cuda::std::array<double, 3> const& buffer)
  {
    return {buffer[0], static_cast<BinCountT>(buffer[1]), buffer[2]};
  }
  HDI WeightedRegressionBin& operator+=(const WeightedRegressionBin& b)
  {
    label_sum += b.label_sum;
    count += b.count;
    weight += b.weight;
    return *this;
  }
  HDI WeightedRegressionBin operator+(WeightedRegressionBin b) const
  {
    b += *this;
    return b;
  }
};
}  // namespace DT
}  // namespace ML
