/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

/** @file internal.h cuML-internal interface to Forest Inference Library. */

#pragma once

#include <cuml/cuml.hpp>

namespace ML {
namespace fil {

/**
 * output_t are flags that define the output produced by the FIL predictor; a
 * valid output_t values consists of the following, combined using '|' (bitwise
 * or), which define stages, which operation in the next stage applied to the
 * output of the previous stage:
 * - one of RAW or AVG, indicating how to combine individual tree outputs into the forest output
 * - optional SIGMOID for applying the sigmoid transform
 * - optional CLASS, to output the class label
 */
enum output_t {
  /** raw output: the sum of the tree outputs; use for GBM models for
      regression, or for binary classification for the value before the
      transformation; note that this value is 0, and may be omitted
      when combined with other flags */
  RAW = 0x0,
  /** average output: divide the sum of the tree outputs by the number of trees
      before further transformations; use for random forests for regression
      and binary classification for the probability */
  AVG = 0x1,
  /** sigmoid transformation: apply 1/(1+exp(-x)) to the sum or average of tree
      outputs; use for GBM binary classification models for probability */
  SIGMOID = 0x10,
  /** output class label: either apply threshold to the output of the previous stage (for binary classification),
      or select the class with the most votes to get the class label (for multi-class classification).  */
  CLASS = 0x100,
  SIGMOID_CLASS = SIGMOID | CLASS,
  AVG_CLASS = AVG | CLASS,
  AVG_SIGMOID_CLASS = AVG | SIGMOID | CLASS,
};

/** val_t is the payload within a FIL leaf */
union val_t {
  /** threshold value for branch node or output value (e.g. class
      probability or regression summand) for leaf node */
  float f;
  /** class label */
  int idx;
};

/** dense_node_t is a node in a densely-stored forest */
struct dense_node_t {
  val_t val;
  int bits;
};

/** sparse_node16_extra_data is what's missing from a dense node to store
    a sparse node, that is, extra indexing information due to compressing
    a sparse tree. */
struct sparse_node16_extra_data {
  int left_idx;
  int dummy;  // make alignment explicit and reserve for future use
};

/** sparse_node16_t is a 16-byte node in a sparsely-stored forest */
struct sparse_node16_t : dense_node_t, sparse_node16_extra_data {
  sparse_node16_t() = default;
  sparse_node16_t(dense_node_t dn, sparse_node16_extra_data ed)
    : dense_node_t(dn), sparse_node16_extra_data(ed) {}
};

/** sparse_node8_t is a node of reduced size (8 bytes)
    in a sparsely-stored forest */
struct sparse_node8_t : dense_node_t {
  sparse_node8_t() = default;
  sparse_node8_t(dense_node_t dn) : dense_node_t(dn) {}
};

/** leaf_algo_t describes what the leaves in a FIL forest store (predict)
    and how FIL aggregates them into class margins/regression result/best class
**/
enum leaf_algo_t {
  /** storing a class probability or regression summand. We add all margins
      together and determine regression result or use threshold to determine
      one of the two classes. **/
  FLOAT_UNARY_BINARY = 0,
  /** storing a class label. Trees vote on the resulting class.
      Probabilities are just normalized votes. */
  CATEGORICAL_LEAF = 1,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      **/
  GROVE_PER_CLASS = 2,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _FEW_CLASSES means fewer (or as many) classes than threads. **/
  GROVE_PER_CLASS_FEW_CLASSES = 3,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _MANY_CLASSES means more classes than threads. **/
  GROVE_PER_CLASS_MANY_CLASSES = 4,
  // to be extended
};

template <leaf_algo_t leaf_algo>
struct leaf_output_t {};
template <>
struct leaf_output_t<leaf_algo_t::FLOAT_UNARY_BINARY> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::CATEGORICAL_LEAF> {
  typedef int T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES> {
  typedef float T;
};

/** node_init initializes node from paramters */
void node_init(dense_node_t* n, val_t output, float thresh, int fid,
               bool def_left, bool is_leaf);
void node_init(sparse_node16_t* node, val_t output, float thresh, int fid,
               bool def_left, bool is_leaf, int left_index);
void node_init(sparse_node8_t* node, val_t output, float thresh, int fid,
               bool def_left, bool is_leaf, int left_index);

/** node_decode extracts individual members from node */
void node_decode(const dense_node_t* node, val_t* output, float* thresh,
                 int* fid, bool* def_left, bool* is_leaf);
void node_decode(const sparse_node16_t* node, val_t* output, float* thresh,
                 int* fid, bool* def_left, bool* is_leaf, int* left_index);
void node_decode(const sparse_node8_t* node, val_t* output, float* thresh,
                 int* fid, bool* def_left, bool* is_leaf, int* left_index);

/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // total number of nodes; ignored for dense forests
  int num_nodes;
  // maximum depth; ignored for sparse forests
  int depth;
  // ntrees is the number of trees
  int num_trees;
  // num_cols is the number of columns in the data
  int num_cols;
  // leaf_algo determines what the leaves store (predict)
  leaf_algo_t leaf_algo;
  // algo is the inference algorithm;
  // sparse forests do not distinguish between NAIVE and TREE_REORG
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if leaf_algo == FLOAT_UNARY_BINARY && (output & OUTPUT_CLASS) != 0 && !predict_proba,
  // and is ignored otherwise
  float threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  float global_bias;
  // only used for CATEGORICAL_LEAF inference. since we're storing the
  // labels in leaves instead of the whole vector, this keeps track
  // of the number of classes
  int num_classes;
  // blocks_per_sm, if nonzero, works as a limit to improve cache hit rate for larger forests
  // suggested values (if nonzero) are from 2 to 7
  // if zero, launches ceildiv(num_rows, NITEMS) blocks
  int blocks_per_sm;
};

/** init_dense uses params and nodes to initialize the dense forest stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param nodes nodes for the forest, of length
      (2**(params->depth + 1) - 1) * params->ntrees
 *  @param params pointer to parameters used to initialize the forest
 */
void init_dense(const raft::handle_t& h, forest_t* pf,
                const dense_node_t* nodes, const forest_params_t* params);

/** init_sparse uses params, trees and nodes to initialize the sparse forest
 *  with 16-byte nodes stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param trees indices of tree roots in the nodes arrray, of length params->ntrees
 *  @param nodes nodes for the forest, of length params->num_nodes
 *  @param params pointer to parameters used to initialize the forest
 */
void init_sparse(const raft::handle_t& h, forest_t* pf, const int* trees,
                 const sparse_node16_t* nodes, const forest_params_t* params);

/** init_sparse uses params, trees and nodes to initialize the sparse forest
 *  with 8-byte nodes stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param trees indices of tree roots in the nodes arrray, of length params->ntrees
 *  @param nodes nodes for the forest, of length params->num_nodes
 *  @param params pointer to parameters used to initialize the forest
 */
void init_sparse(const raft::handle_t& h, forest_t* pf, const int* trees,
                 const sparse_node8_t* nodes, const forest_params_t* params);

}  // namespace fil
}  // namespace ML
