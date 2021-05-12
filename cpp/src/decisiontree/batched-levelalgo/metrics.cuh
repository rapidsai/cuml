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

struct Int2Max {
  DI int2 operator()(const int2& a, const int2& b) {
    int2 out;
    if (a.y > b.y)
      out = a;
    else if (a.y == b.y && a.x < b.x)
      out = a;
    else
      out = b;
    return out;
  }
};  // struct Int2Max

/**
   * @note to be called by only one block from all participating blocks
   *       'smem' must be atleast of size `sizeof(int) * input.nclasses`
   */
template <typename IdxT, typename LabelT, typename DataT, int TPB>
static DI void ComputeClassificationPrediction(
  IdxT range_start, IdxT range_len, const Input<DataT, LabelT, IdxT>& input,
  volatile Node<DataT, LabelT, IdxT>* nodes, IdxT* n_leaves, void* smem) {
  typedef cub::BlockReduce<int2, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;
  auto* shist = reinterpret_cast<int*>(smem);
  auto tid = threadIdx.x;
  for (int i = tid; i < input.nclasses; i += blockDim.x) shist[i] = 0;
  __syncthreads();
  auto len = range_start + range_len;
  for (auto i = range_start + tid; i < len; i += blockDim.x) {
    auto label = input.labels[input.rowids[i]];
    atomicAdd(shist + int(label), 1);
  }
  __syncthreads();
  auto op = Int2Max();
  int2 v = {-1, -1};
  for (int i = tid; i < input.nclasses; i += blockDim.x) {
    int2 tmp = {i, shist[i]};
    v = op(v, tmp);
  }
  v = BlockReduceT(temp).Reduce(v, op);
  __syncthreads();
  if (tid == 0) {
    nodes[0].makeLeaf(n_leaves, LabelT(v.x));
  }
}

/**
   * @note to be called by only one block from all participating blocks
   *       'smem' is not used, but kept for the sake of interface parity with
   *       the corresponding method for classification
   */
template <typename IdxT, typename LabelT, typename DataT, int TPB>
static DI void ComputeRegressionPrediction(
  IdxT range_start, IdxT range_len, const Input<DataT, LabelT, IdxT>& input,
  volatile Node<DataT, LabelT, IdxT>* nodes, IdxT* n_leaves, void* smem) {
  typedef cub::BlockReduce<LabelT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;
  LabelT sum = LabelT(0.0);
  auto tid = threadIdx.x;
  auto len = range_start + range_len;
  for (auto i = range_start + tid; i < len; i += blockDim.x) {
    auto label = input.labels[input.rowids[i]];
    sum += label;
  }
  sum = BlockReduceT(temp).Sum(sum);
  __syncthreads();
  if (tid == 0) {
    if (range_len != 0) {
      nodes[0].makeLeaf(n_leaves, sum / range_len);
    } else {
      nodes[0].makeLeaf(n_leaves, 0.0);
    }
  }
}

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
  template <int TPB>
  static DI void computePrediction(IdxT range_start, IdxT range_len,
                                   const Input<DataT, LabelT, IdxT>& input,
                                   volatile Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT* n_leaves, void* smem) {
    ComputeClassificationPrediction<IdxT, LabelT, DataT, TPB>(
      range_start, range_len, input, nodes, n_leaves, smem);
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
  template <int TPB>
  static DI void computePrediction(IdxT range_start, IdxT range_len,
                                   const Input<DataT, LabelT, IdxT>& input,
                                   volatile Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT* n_leaves, void* smem) {
    ComputeClassificationPrediction<IdxT, LabelT, DataT, TPB>(
      range_start, range_len, input, nodes, n_leaves, smem);
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

    DI static void AtomicAdd(MSEBin* hist, int nbins, int b, double label) {
      atomicAdd(&(hist + b)->label_sum, label);
      atomicAdd(&(hist + b)->count, 1);
    }
    DI static void AtomicAddGlobal(MSEBin* address, MSEBin val) {
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
        gain = -NumericLimits<DataT>::kMax;
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
        gain = -NumericLimits<DataT>::kMax;
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
  template <int TPB>
  static DI void computePrediction(IdxT range_start, IdxT range_len,
                                   const Input<DataT, LabelT, IdxT>& input,
                                   volatile Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT* n_leaves, void* smem) {
    ComputeRegressionPrediction<IdxT, LabelT, DataT, TPB>(
      range_start, range_len, input, nodes, n_leaves, smem);
  }
};

}  // namespace DecisionTree
}  // namespace ML
