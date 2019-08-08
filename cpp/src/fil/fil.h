/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <treelite/c_api.h>
#include "cuML.hpp"

namespace ML {
namespace fil {

/** @note FIL only supports inference with single precision.
 *  TODO(canonizer): parameterize the functions and structures by the data type
 *  and the threshold/weight type.
 */

/** Inference algorithm to use. */
enum algo_t {
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

/** 
 * output_t are flags that define the output produced by the FIL predictor; a
 * valid output_t values consists of the following, combined using '|' (bitwise
 * or), which define stages, which operation in the next stage applied to the
 * output of the previous stage:
 * - one of RAW or AVG, indicating how to combine individual tree outputs into the forest output
 * - optional SIGMOID for applying the sigmoid transform
 * - optional THRESHOLD, for thresholding for classification
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
  /** threshold: apply threshold to the output of the previous stage to get the
      class (0 or 1) */
  THRESHOLD = 0x100,
};

/** dense_node is a single tree node */
struct dense_node_t {
  float val;
  int bits;
};

/** dense_node_init initializes n from paramters */
void dense_node_init(dense_node_t* n, float output, float thresh, int fid,
                     bool def_left, bool is_leaf);

/** dense_node_decode extracts individual members from n */
void dense_node_decode(const dense_node_t* n, float* output, float* thresh,
                       int* fid, bool* def_left, bool* is_leaf);

struct forest;

/** forest_t is the predictor handle */
typedef forest* forest_t;

/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // nodes of trees, of length (2**(depth + 1) - 1) * ntrees
  const dense_node_t* nodes;
  // depth of each tree
  int depth;
  // ntrees is the number of trees
  int ntrees;
  // cols is the number of columns in the data
  int cols;
  // algo is the inference algorithm
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if output == OUTPUT_CLASS,
  // and is ignored otherwise
  float threshold;
};

/** treelite_params_t are parameters for importing treelite models */
struct treelite_params_t {
  // algo is the inference algorithm
  algo_t algo;
  // output_class indicates whether thresholding will be applied
  // to the model output
  bool output_class;
  // threshold is used for thresholding if output_class == true,
  // and is ignored otherwise
  float threshold;
};

/** init_dense uses params to initialize the forest stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param params pointer to parameters used to initialize the forest
 */
void init_dense(const cumlHandle& h, forest_t* pf,
                const forest_params_t* params);

/** from_treelite uses a treelite model to initialize the forest
 * @param handle cuML handle used by this function
 * @param pforest pointer to where to store the newly created forest
 * @param model treelite model used to initialize the forest
 * @param tl_params additional parameters for the forest
 */
void from_treelite(const cumlHandle& handle, forest_t* pforest,
                   ModelHandle model, const treelite_params_t* tl_params);

/** free deletes forest and all resources held by it; after this, forest is no longer usable 
 *  @param h cuML handle used by this function
 *  @param f the forest to free; not usable after the call to this function
 */
void free(const cumlHandle& h, forest_t f);

/** predict predicts on data (with n rows) using forest and writes results into preds;
 *  the number of columns is stored in forest, and both preds and data point to GPU memory
 *  @param h cuML handle used by this function
 *  @param f forest used for predictions
 *  @param preds array of size n in GPU memory to store predictions into
 *  @param data array of size n * cols (cols is the number of columns 
 *      for the forest f) from which to predict
 *  @param n number of data rows
 */
void predict(const cumlHandle& h, forest_t f, float* preds, const float* data,
             size_t n);

}  // namespace fil
}  // namespace ML
