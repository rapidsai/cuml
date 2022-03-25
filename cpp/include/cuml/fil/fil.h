/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/** @file fil.h Interface to the forest inference library. */

#pragma once

#include <stddef.h>

#include <cuml/ensemble/treelite_defs.hpp>

namespace raft {
class handle_t;
}

namespace ML {
namespace fil {

/** @note FIL only supports inference with single precision.
 *  TODO(canonizer): parameterize the functions and structures by the data type
 *  and the threshold/weight type.
 */

/** Inference algorithm to use. */
enum algo_t {
  /** choose the algorithm automatically; currently chooses NAIVE for sparse forests
      and BATCH_TREE_REORG for dense ones */
  ALGO_AUTO,
  /** naive algorithm: 1 thread block predicts 1 row; the row is cached in
      shared memory, and the trees are distributed cyclically between threads */
  NAIVE,
  /** tree reorg algorithm: same as naive, but the tree nodes are rearranged
      into a more coalescing-friendly layout: for every node position,
      nodes of all trees at that position are stored next to each other */
  TREE_REORG,
  /** batch tree reorg algorithm: same as tree reorg, but predictions multiple rows (up to 4)
      in a single thread block */
  BATCH_TREE_REORG
};

/** storage_type_t defines whether to import the forests as dense or sparse */
enum storage_type_t {
  /** decide automatically; currently always builds dense forests */
  AUTO,
  /** import the forest as dense (8 or 16-bytes nodes, depending on model precision */
  DENSE,
  /** import the forest as sparse (currently always with 16-byte nodes) */
  SPARSE,
  /** (experimental) import the forest as sparse with 8-byte nodes; can fail if
      8-byte nodes are not enough to store the forest, e.g. there are too many
      nodes in a tree or too many features or the thresholds are double precision;
      note that the number of bits used to store the child or feature index can
      change in the future; this can affect whether a particular forest can be
      imported as SPARSE8 */
  SPARSE8,
};
static const char* storage_type_repr[] = {"AUTO", "DENSE", "SPARSE", "SPARSE8"};

template <typename real_t>
struct forest;

/** forest_t is the predictor handle */
template <typename real_t>
using forest_t = forest<real_t>*;

/** MAX_N_ITEMS determines the maximum allowed value for tl_params::n_items */
constexpr int MAX_N_ITEMS = 4;

/** treelite_params_t are parameters for importing treelite models */
struct treelite_params_t {
  // algo is the inference algorithm
  algo_t algo;
  // output_class indicates whether thresholding will be applied
  // to the model output
  bool output_class;
  // threshold may be used for thresholding if output_class == true,
  // and is ignored otherwise. threshold is ignored if leaves store
  // vectorized class labels. in that case, a class with most votes
  // is returned regardless of the absolute vote count
  float threshold;
  // storage_type indicates whether the forest should be imported as dense or sparse
  storage_type_t storage_type;
  // blocks_per_sm, if nonzero, works as a limit to improve cache hit rate for larger forests
  // suggested values (if nonzero) are from 2 to 7
  // if zero, launches ceildiv(num_rows, NITEMS) blocks
  int blocks_per_sm;
  // threads_per_tree determines how many threads work on a single tree at once inside a block
  // can only be a power of 2
  int threads_per_tree;
  // n_items is how many input samples (items) any thread processes. If 0 is given,
  // choose most (up to MAX_N_ITEMS) that fit into shared memory.
  int n_items;
  // if non-nullptr, *pforest_shape_str will be set to caller-owned string that
  // contains forest shape
  char** pforest_shape_str;
};

/** from_treelite uses a treelite model to initialize the forest
 * @param handle cuML handle used by this function
 * @param pforest pointer to where to store the newly created forest
 * @param model treelite model used to initialize the forest
 * @param tl_params additional parameters for the forest
 */
// TODO (canonizer): use std::variant<forest_t<float> forest_t<double>>* for pforest
void from_treelite(const raft::handle_t& handle,
                   forest_t<float>* pforest,
                   ModelHandle model,
                   const treelite_params_t* tl_params);

/** free deletes forest and all resources held by it; after this, forest is no longer usable
 *  @param h cuML handle used by this function
 *  @param f the forest to free; not usable after the call to this function
 */
template <typename real_t>
void free(const raft::handle_t& h, forest_t<real_t> f);

/** predict predicts on data (with n rows) using forest and writes results into preds;
 *  the number of columns is stored in forest, and both preds and data point to GPU memory
 *  @param h cuML handle used by this function
 *  @param f forest used for predictions
 *  @param preds array in GPU memory to store predictions into
 *      size = predict_proba ? (2*num_rows) : num_rows
 *  @param data array of size n * cols (cols is the number of columns
 *      for the forest f) from which to predict
 *  @param num_rows number of data rows
 *  @param predict_proba for classifier models, this forces to output both class probabilities
 *      instead of binary class prediction. format matches scikit-learn API
 */
template <typename real_t>
void predict(const raft::handle_t& h,
             forest_t<real_t> f,
             real_t* preds,
             const real_t* data,
             size_t num_rows,
             bool predict_proba = false);

}  // namespace fil
}  // namespace ML
