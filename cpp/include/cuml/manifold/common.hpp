/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <stdint.h>

namespace ML {

// Dense input uses int64_t until FAISS is updated
typedef int64_t knn_indices_dense_t;

typedef int knn_indices_sparse_t;

/**
 * Simple container for KNN graph properties
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct knn_graph {
  knn_graph(value_idx n_rows_, int n_neighbors_)
    : n_rows(n_rows_), n_neighbors(n_neighbors_), knn_indices{nullptr}, knn_dists{nullptr}
  {
  }

  knn_graph(value_idx n_rows_, int n_neighbors_, value_idx* knn_indices_, value_t* knn_dists_)
    : n_rows(n_rows_), n_neighbors(n_neighbors_), knn_indices(knn_indices_), knn_dists(knn_dists_)
  {
  }

  value_idx* knn_indices;
  value_t* knn_dists;

  value_idx n_rows;
  int n_neighbors;
};

/**
 * Base struct for representing inputs to manifold learning
 * algorithms.
 * @tparam T
 */
template <typename T>
struct manifold_inputs_t {
  T* y;
  int n;
  int d;

  manifold_inputs_t(T* y_, int n_, int d_) : y(y_), n(n_), d(d_) {}

  virtual bool alloc_knn_graph() const = 0;
};

/**
 * Dense input to manifold learning algorithms
 * @tparam T
 */
template <typename T>
struct manifold_dense_inputs_t : public manifold_inputs_t<T> {
  T* X;

  manifold_dense_inputs_t(T* x_, T* y_, int n_, int d_) : manifold_inputs_t<T>(y_, n_, d_), X(x_) {}

  bool alloc_knn_graph() const { return true; }
};

/**
 * Sparse CSR input to manifold learning algorithms
 * @tparam value_idx
 * @tparam T
 */
template <typename value_idx, typename T>
struct manifold_sparse_inputs_t : public manifold_inputs_t<T> {
  value_idx* indptr;
  value_idx* indices;
  T* data;

  size_t nnz;

  manifold_sparse_inputs_t(
    value_idx* indptr_, value_idx* indices_, T* data_, T* y_, size_t nnz_, int n_, int d_)
    : manifold_inputs_t<T>(y_, n_, d_), indptr(indptr_), indices(indices_), data(data_), nnz(nnz_)
  {
  }

  bool alloc_knn_graph() const { return true; }
};

/**
 * Precomputed KNN graph input to manifold learning algorithms
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct manifold_precomputed_knn_inputs_t : public manifold_inputs_t<value_t> {
  manifold_precomputed_knn_inputs_t<value_idx, value_t>(
    value_idx* knn_indices_, value_t* knn_dists_, value_t* y_, int n_, int d_, int n_neighbors_)
    : manifold_inputs_t<value_t>(y_, n_, d_), knn_graph(n_, n_neighbors_, knn_indices_, knn_dists_)
  {
  }

  knn_graph<value_idx, value_t> knn_graph;

  bool alloc_knn_graph() const { return false; }
};

};  // end namespace ML
