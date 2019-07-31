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
#pragma once
#include "levelkernel_classifier.cuh"
template <typename T, typename F>
void initial_metric_classification(
  int *labels, unsigned int *sample_cnt, const int nrows,
  const int n_unique_labels, std::vector<int> &histvec, T &initial_metric,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  CUDA_CHECK(cudaMemsetAsync(tempmem->d_parent_hist->data(), 0,
                             n_unique_labels * sizeof(unsigned int),
                             tempmem->stream));
  int blocks = MLCommon::ceildiv(nrows, 128);
  gini_kernel_level<<<blocks, 128, sizeof(int) * n_unique_labels,
                      tempmem->stream>>>(labels, sample_cnt, nrows,
                                         n_unique_labels,
                                         (int *)tempmem->d_parent_hist->data());
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(tempmem->h_parent_hist->data(),
                       tempmem->d_parent_hist->data(), n_unique_labels,
                       tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  histvec.assign(tempmem->h_parent_hist->data(),
                 tempmem->h_parent_hist->data() + n_unique_labels);
  initial_metric = F::exec(histvec, nrows);
}

template <typename T>
void get_histogram_classification(
  T *data, int *labels, unsigned int *flags, unsigned int *sample_cnt,
  const int nrows, const int ncols, const int n_unique_labels, const int nbins,
  const int n_nodes, std::shared_ptr<TemporaryMemory<T, int>> tempmem,
  unsigned int *histout) {
  size_t histcount = ncols * nbins * n_unique_labels * n_nodes;
  CUDA_CHECK(cudaMemsetAsync(histout, 0, histcount * sizeof(unsigned int),
                             tempmem->stream));
  int node_batch = min(n_nodes, tempmem->max_nodes_class);
  size_t shmem = nbins * n_unique_labels * sizeof(int) * node_batch;
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);

  if ((n_nodes == node_batch)) {
    get_hist_kernel<<<blocks, threads, shmem, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      n_unique_labels, nbins, n_nodes, tempmem->d_quantile->data(), histout);
  } else {
    get_hist_kernel_global<<<blocks, threads, 0, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      n_unique_labels, nbins, n_nodes, tempmem->d_quantile->data(), histout);
  }
  CUDA_CHECK(cudaGetLastError());
}
template <typename T, typename F, typename DF>
void get_best_split_classification(
  unsigned int *hist, unsigned int *d_hist,
  const std::vector<unsigned int> &colselector, unsigned int *d_colids,
  const int nbins, const int n_unique_labels, const int n_nodes,
  const int depth, const int min_rpn, float *gain,
  std::vector<std::vector<int>> &sparse_histstate,
  std::vector<SparseTreeNode<T, int>> &sparsetree, const int sparsesize,
  std::vector<int> &sparse_nodelist, int *split_colidx, int *split_binidx,
  int *d_split_colidx, int *d_split_binidx,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  T *quantile = tempmem->h_quantile->data();
  int ncols = colselector.size();
  size_t histcount = ncols * nbins * n_unique_labels * n_nodes;
  bool use_gpu_flag = false;
  if (n_nodes > 512) use_gpu_flag = true;
  memset(gain, 0, n_nodes * sizeof(float));

  int sparsetree_sz = sparsetree.size();
  if (use_gpu_flag) {
    //GPU based best split
    unsigned int *h_parent_hist, *d_parent_hist, *d_child_hist, *h_child_hist;
    T *d_parent_metric, *d_child_best_metric;
    T *h_parent_metric, *h_child_best_metric;
    float *d_outgain, *h_outgain;
    h_parent_hist = tempmem->h_parent_hist->data();
    h_child_hist = tempmem->h_child_hist->data();
    h_parent_metric = tempmem->h_parent_metric->data();
    h_child_best_metric = tempmem->h_child_best_metric->data();
    h_outgain = tempmem->h_outgain->data();

    d_parent_hist = tempmem->d_parent_hist->data();
    d_child_hist = tempmem->d_child_hist->data();
    d_parent_metric = tempmem->d_parent_metric->data();
    d_child_best_metric = tempmem->d_child_best_metric->data();
    d_outgain = tempmem->d_outgain->data();
    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int sparse_nodeid = sparse_nodelist[nodecnt];
      int parentid = sparsesize + sparse_nodeid;
      std::vector<int> &parent_hist = sparse_histstate[parentid];
      h_parent_metric[nodecnt] = sparsetree[parentid].best_metric_val;
      for (int j = 0; j < n_unique_labels; j++) {
        h_parent_hist[nodecnt * n_unique_labels + j] = parent_hist[j];
      }
    }

    MLCommon::updateDevice(d_parent_hist, h_parent_hist,
                           n_nodes * n_unique_labels, tempmem->stream);
    MLCommon::updateDevice(d_parent_metric, h_parent_metric, n_nodes,
                           tempmem->stream);
    int threads = 64;
    size_t shmemsz = (threads + 2) * 2 * n_unique_labels * sizeof(int);
    get_best_split_classification_kernel<T, DF>
      <<<n_nodes, threads, shmemsz, tempmem->stream>>>(
        d_hist, d_parent_hist, d_parent_metric, d_colids, nbins, ncols, n_nodes,
        n_unique_labels, min_rpn, d_outgain, d_split_colidx, d_split_binidx,
        d_child_hist, d_child_best_metric);
    CUDA_CHECK(cudaGetLastError());
    MLCommon::updateHost(h_child_hist, d_child_hist,
                         2 * n_nodes * n_unique_labels, tempmem->stream);
    MLCommon::updateHost(h_outgain, d_outgain, n_nodes, tempmem->stream);
    MLCommon::updateHost(h_child_best_metric, d_child_best_metric, 2 * n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(split_binidx, d_split_binidx, n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(split_colidx, d_split_colidx, n_nodes,
                         tempmem->stream);

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int sparse_nodeid = sparse_nodelist[nodecnt];
      std::vector<int> tmp_histleft(n_unique_labels);
      std::vector<int> tmp_histright(n_unique_labels);
      for (int j = 0; j < n_unique_labels; j++) {
        tmp_histleft[j] = h_child_hist[n_unique_labels * nodecnt * 2 + j];
        tmp_histright[j] =
          h_child_hist[n_unique_labels * nodecnt * 2 + j + n_unique_labels];
      }
      //Sparse tree
      SparseTreeNode<T, int> &curr_node =
        sparsetree[sparsesize + sparse_nodeid];
      curr_node.colid = split_colidx[nodecnt];
      curr_node.quesval =
        quantile[split_colidx[nodecnt] * nbins + split_binidx[nodecnt]];
      curr_node.left_child_id = sparsetree_sz + 2 * nodecnt;
      SparseTreeNode<T, int> leftnode, rightnode;
      leftnode.best_metric_val = h_child_best_metric[2 * nodecnt];
      rightnode.best_metric_val = h_child_best_metric[2 * nodecnt + 1];
      sparsetree.push_back(leftnode);
      sparsetree.push_back(rightnode);
      sparse_histstate[curr_node.left_child_id] = tmp_histleft;
      sparse_histstate[curr_node.left_child_id + 1] = tmp_histright;
    }
  } else {
    MLCommon::updateHost(hist, d_hist, histcount, tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      std::vector<T> bestmetric(2, 0);
      int nodeoffset = nodecnt * nbins * n_unique_labels;
      int sparse_nodeid = sparse_nodelist[nodecnt];
      int parentid = sparsesize + sparse_nodeid;
      int best_col_id = -1;
      int best_bin_id = -1;
      std::vector<int> besthist_left(n_unique_labels);
      std::vector<int> besthist_right(n_unique_labels);

      for (int colid = 0; colid < ncols; colid++) {
        int coloffset = colid * nbins * n_unique_labels * n_nodes;
        for (int binid = 0; binid < nbins; binid++) {
          int binoffset = binid * n_unique_labels;
          int tmp_lnrows = 0;
          int tmp_rnrows = 0;
          std::vector<int> tmp_histleft(n_unique_labels, 0);
          std::vector<int> tmp_histright(n_unique_labels, 0);

          std::vector<int> &parent_hist = sparse_histstate[parentid];
          // Compute gini right and gini left value for each bin.
          for (int j = 0; j < n_unique_labels; j++) {
            tmp_histleft[j] = hist[coloffset + binoffset + nodeoffset + j];
            tmp_histright[j] = parent_hist[j] - tmp_histleft[j];
          }
          for (int j = 0; j < n_unique_labels; j++) {
            tmp_lnrows += tmp_histleft[j];
            tmp_rnrows += tmp_histright[j];
          }
          int totalrows = tmp_lnrows + tmp_rnrows;
          if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows < min_rpn)
            continue;

          float tmp_gini_left = F::exec(tmp_histleft, tmp_lnrows);
          float tmp_gini_right = F::exec(tmp_histright, tmp_rnrows);

          float max_value = F::max_val(n_unique_labels);

          ASSERT((tmp_gini_left >= 0.0f) && (tmp_gini_left <= max_value),
                 "gini left value %f not in [0.0, %f]", tmp_gini_left,
                 max_value);
          ASSERT((tmp_gini_right >= 0.0f) && (tmp_gini_right <= max_value),
                 "gini right value %f not in [0.0, %f]", tmp_gini_right,
                 max_value);

          float impurity = (tmp_lnrows * 1.0f / totalrows) * tmp_gini_left +
                           (tmp_rnrows * 1.0f / totalrows) * tmp_gini_right;
          float info_gain = sparsetree[parentid].best_metric_val - impurity;

          // Compute best information col_gain so far
          if (info_gain > gain[nodecnt]) {
            gain[nodecnt] = info_gain;
            best_bin_id = binid;
            best_col_id = colselector[colid];
            besthist_left = tmp_histleft;
            besthist_right = tmp_histright;
            bestmetric[0] = tmp_gini_left;
            bestmetric[1] = tmp_gini_right;
          }
        }
      }
      split_colidx[nodecnt] = best_col_id;
      split_binidx[nodecnt] = best_bin_id;
      //Sparse tree
      SparseTreeNode<T, int> &curr_node =
        sparsetree[sparsesize + sparse_nodeid];
      curr_node.colid = split_colidx[nodecnt];
      curr_node.quesval =
        quantile[split_colidx[nodecnt] * nbins + split_binidx[nodecnt]];
      curr_node.left_child_id = sparsetree_sz + 2 * nodecnt;
      SparseTreeNode<T, int> leftnode, rightnode;
      leftnode.best_metric_val = bestmetric[0];
      rightnode.best_metric_val = bestmetric[1];
      sparsetree.push_back(leftnode);
      sparsetree.push_back(rightnode);
      sparse_histstate[curr_node.left_child_id] = besthist_left;
      sparse_histstate[curr_node.left_child_id + 1] = besthist_right;
    }
    MLCommon::updateDevice(d_split_binidx, split_binidx, n_nodes,
                           tempmem->stream);
    MLCommon::updateDevice(d_split_colidx, split_colidx, n_nodes,
                           tempmem->stream);
  }
}

template <typename T>
void leaf_eval_classification(
  float *gain, int curr_depth, const int max_depth, const int max_leaves,
  unsigned int *new_node_flags, std::vector<SparseTreeNode<T, int>> &sparsetree,
  const int sparsesize, std::vector<std::vector<int>> &sparse_hist,
  int &n_nodes_next, std::vector<int> &sparse_nodelist, int &tree_leaf_cnt) {
  std::vector<int> tmp_sparse_nodelist(sparse_nodelist);
  sparse_nodelist.clear();

  int non_leaf_counter = 0;
  bool condition_global = (curr_depth == max_depth);
  if (max_leaves != -1)
    condition_global = condition_global || (tree_leaf_cnt >= max_leaves);

  for (int i = 0; i < tmp_sparse_nodelist.size(); i++) {
    unsigned int node_flag;
    int sparse_nodeid = tmp_sparse_nodelist[i];
    std::vector<int> &nodehist = sparse_hist[sparsesize + sparse_nodeid];
    bool condition = condition_global || (gain[i] == 0.0);
    if (condition) {
      node_flag = 0xFFFFFFFF;
      sparsetree[sparsesize + sparse_nodeid].colid = -1;
      sparsetree[sparsesize + sparse_nodeid].prediction =
        get_class_hist(nodehist);
    } else {
      sparse_nodelist.push_back(2 * i);
      sparse_nodelist.push_back(2 * i + 1);
      node_flag = non_leaf_counter;
      non_leaf_counter++;
    }
    new_node_flags[i] = node_flag;
  }
  int nleafed = tmp_sparse_nodelist.size() - non_leaf_counter;
  tree_leaf_cnt += nleafed;
  n_nodes_next = 2 * non_leaf_counter;
}
