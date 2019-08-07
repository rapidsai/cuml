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
#include "levelkernel_regressor.cuh"
template <typename T, typename F>
void initial_metric_regression(const T *labels, unsigned int *sample_cnt,
                               const int nrows, T &mean, unsigned int &count,
                               T &initial_metric,
                               std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  CUDA_CHECK(
    cudaMemsetAsync(tempmem->d_mseout->data(), 0, sizeof(T), tempmem->stream));
  CUDA_CHECK(
    cudaMemsetAsync(tempmem->d_predout->data(), 0, sizeof(T), tempmem->stream));
  CUDA_CHECK(cudaMemsetAsync(tempmem->d_count->data(), 0, sizeof(unsigned int),
                             tempmem->stream));
  int threads = 128;
  int blocks = MLCommon::ceildiv(nrows, threads);

  pred_kernel_level<<<blocks, threads, 0, tempmem->stream>>>(
    labels, sample_cnt, nrows, tempmem->d_predout->data(),
    tempmem->d_count->data());
  CUDA_CHECK(cudaGetLastError());
  mse_kernel_level<T, F><<<blocks, threads, 0, tempmem->stream>>>(
    labels, sample_cnt, nrows, tempmem->d_predout->data(),
    tempmem->d_count->data(), tempmem->d_mseout->data());
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(tempmem->h_count->data(), tempmem->d_count->data(), 1,
                       tempmem->stream);
  MLCommon::updateHost(tempmem->h_predout->data(), tempmem->d_predout->data(),
                       1, tempmem->stream);
  MLCommon::updateHost(tempmem->h_mseout->data(), tempmem->d_mseout->data(), 1,
                       tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  count = tempmem->h_count->data()[0];
  mean = tempmem->h_predout->data()[0] / count;
  initial_metric = tempmem->h_mseout->data()[0] / count;
}

template <typename T, typename F>
void get_mse_regression(const T *data, const T *labels, unsigned int *flags,
                        unsigned int *sample_cnt, const int nrows,
                        const int ncols, const int nbins, const int n_nodes,
                        std::shared_ptr<TemporaryMemory<T, T>> tempmem,
                        T *d_mseout, T *d_predout, unsigned int *d_count) {
  size_t predcount = ncols * nbins * n_nodes;
  CUDA_CHECK(
    cudaMemsetAsync(d_mseout, 0, 2 * predcount * sizeof(T), tempmem->stream));
  CUDA_CHECK(
    cudaMemsetAsync(d_predout, 0, predcount * sizeof(T), tempmem->stream));
  CUDA_CHECK(cudaMemsetAsync(d_count, 0, predcount * sizeof(unsigned int),
                             tempmem->stream));

  int node_batch_pred = min(n_nodes, tempmem->max_nodes_pred);
  int node_batch_mse = min(n_nodes, tempmem->max_nodes_mse);
  size_t shmempred = nbins * (sizeof(unsigned int) + sizeof(T)) * n_nodes;
  size_t shmemmse = shmempred + 2 * nbins * n_nodes * sizeof(T);

  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);

  if ((n_nodes == node_batch_pred)) {
    get_pred_kernel<T, QuantileQues<T>>
      <<<blocks, threads, shmempred, tempmem->stream>>>(
        data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows,
        ncols, nbins, n_nodes, tempmem->d_quantile->data(), d_predout, d_count);
  } else {
    get_pred_kernel_global<T, QuantileQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows,
        ncols, nbins, n_nodes, tempmem->d_quantile->data(), d_predout, d_count);
  }
  CUDA_CHECK(cudaGetLastError());
  if ((n_nodes == node_batch_mse)) {
    get_mse_kernel<T, F, QuantileQues<T>>
      <<<blocks, threads, shmemmse, tempmem->stream>>>(
        data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows,
        ncols, nbins, n_nodes, tempmem->d_quantile->data(),
        tempmem->d_parent_pred->data(), tempmem->d_parent_count->data(),
        d_predout, d_count, d_mseout);
  } else {
    get_mse_kernel_global<T, F, QuantileQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows,
        ncols, nbins, n_nodes, tempmem->d_quantile->data(),
        tempmem->d_parent_pred->data(), tempmem->d_parent_count->data(),
        d_predout, d_count, d_mseout);
  }
  CUDA_CHECK(cudaGetLastError());
}
template <typename T>
void get_best_split_regression(
  T *mseout, T *d_mseout, T *predout, T *d_predout, unsigned int *count,
  unsigned int *d_count, const std::vector<unsigned int> &colselector,
  unsigned int *d_colids, const int nbins, const int n_nodes, const int depth,
  const int min_rpn, const int split_algo, const int sparsesize, float *gain,
  std::vector<T> &sparse_meanstate,
  std::vector<unsigned int> &sparse_countstate,
  std::vector<SparseTreeNode<T, T>> &sparsetree,
  std::vector<int> &sparse_nodelist, int *split_colidx, int *split_binidx,
  int *d_split_colidx, int *d_split_binidx,
  std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  T *quantile = tempmem->h_quantile->data();
  int ncols = colselector.size();
  size_t predcount = ncols * nbins * n_nodes;
  bool use_gpu_flag = false;
  if (n_nodes > 512) use_gpu_flag = true;

  memset(gain, 0, n_nodes * sizeof(float));
  int sparsetree_sz = sparsetree.size();
  if (use_gpu_flag) {
    int threads = 64;

    T *h_parentmetric = tempmem->h_parent_metric->data();
    float *h_outgain = tempmem->h_outgain->data();
    T *h_childmean = tempmem->h_child_pred->data();
    unsigned int *h_childcount = tempmem->h_child_count->data();
    T *h_childmetric = tempmem->h_child_best_metric->data();

    T *d_parentmean = tempmem->d_parent_pred->data();
    unsigned int *d_parentcount = tempmem->d_parent_count->data();
    T *d_parentmetric = tempmem->d_parent_metric->data();
    float *d_outgain = tempmem->d_outgain->data();
    T *d_childmean = tempmem->d_child_pred->data();
    unsigned int *d_childcount = tempmem->d_child_count->data();
    T *d_childmetric = tempmem->d_child_best_metric->data();

    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int sparse_nodeid = sparse_nodelist[nodecnt];
      h_parentmetric[nodecnt] =
        sparsetree[sparsesize + sparse_nodeid].best_metric_val;
    }

    //Here parent mean and count are already updated
    MLCommon::updateDevice(d_parentmetric, h_parentmetric, n_nodes,
                           tempmem->stream);

    get_best_split_regression_kernel<<<n_nodes, threads, 0, tempmem->stream>>>(
      d_mseout, d_predout, d_count, d_parentmean, d_parentcount, d_parentmetric,
      d_colids, nbins, ncols, n_nodes, min_rpn, d_outgain, d_split_colidx,
      d_split_binidx, d_childmean, d_childcount, d_childmetric);
    CUDA_CHECK(cudaGetLastError());

    MLCommon::updateHost(h_childmetric, d_childmetric, 2 * n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(h_outgain, d_outgain, n_nodes, tempmem->stream);
    MLCommon::updateHost(h_childmean, d_childmean, 2 * n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(h_childcount, d_childcount, 2 * n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(split_binidx, d_split_binidx, n_nodes,
                         tempmem->stream);
    MLCommon::updateHost(split_colidx, d_split_colidx, n_nodes,
                         tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int sparse_nodeid = sparse_nodelist[nodecnt];
      SparseTreeNode<T, T> &curr_node = sparsetree[sparsesize + sparse_nodeid];
      curr_node.colid = split_colidx[nodecnt];
      T *dummy = nullptr;
      curr_node.quesval =
        getQuesValue(dummy, quantile, nbins, split_colidx[nodecnt],
                     split_binidx[nodecnt], split_algo);

      curr_node.left_child_id = sparsetree_sz + 2 * nodecnt;
      sparse_meanstate[curr_node.left_child_id] = h_childmean[nodecnt * 2];
      sparse_meanstate[curr_node.left_child_id + 1] =
        h_childmean[nodecnt * 2 + 1];
      sparse_countstate[curr_node.left_child_id] = h_childcount[nodecnt * 2];
      sparse_countstate[curr_node.left_child_id + 1] =
        h_childcount[nodecnt * 2 + 1];
      SparseTreeNode<T, T> leftnode, rightnode;
      leftnode.best_metric_val = h_childmetric[nodecnt * 2];
      rightnode.best_metric_val = h_childmetric[nodecnt * 2 + 1];
      sparsetree.push_back(leftnode);
      sparsetree.push_back(rightnode);
    }

  } else {
    MLCommon::updateHost(mseout, d_mseout, 2 * predcount, tempmem->stream);
    MLCommon::updateHost(predout, d_predout, predcount, tempmem->stream);
    MLCommon::updateHost(count, d_count, predcount, tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      T bestmetric_left = 0;
      T bestmetric_right = 0;
      int nodeoff_mse = nodecnt * nbins * 2;
      int nodeoff_pred = nodecnt * nbins;
      int sparse_nodeid = sparse_nodelist[nodecnt];
      int parentid = sparse_nodeid + sparsesize;
      int best_col_id = 0;
      int best_bin_id = 0;
      T bestmean_left = 0;
      T bestmean_right = 0;
      unsigned int bestcount_left = 0;
      unsigned int bestcount_right = 0;
      T parent_mean = sparse_meanstate[parentid];
      unsigned int parent_count = sparse_countstate[parentid];
      for (int colid = 0; colid < ncols; colid++) {
        int coloff_mse = colid * nbins * 2 * n_nodes;
        int coloff_pred = colid * nbins * n_nodes;
        for (int binid = 0; binid < nbins; binid++) {
          int binoff_mse = binid * 2;
          int binoff_pred = binid;
          unsigned int tmp_lnrows = 0;
          unsigned int tmp_rnrows = 0;

          tmp_lnrows = count[coloff_pred + binoff_pred + nodeoff_pred];
          tmp_rnrows = parent_count - tmp_lnrows;
          unsigned int totalrows = tmp_lnrows + tmp_rnrows;
          if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows < min_rpn)
            continue;
          T tmp_meanleft = predout[coloff_pred + binoff_pred + nodeoff_pred];
          T tmp_meanright = parent_mean * parent_count - tmp_meanleft;
          tmp_meanleft /= tmp_lnrows;
          tmp_meanright /= tmp_rnrows;
          T tmp_mse_left =
            mseout[coloff_mse + binoff_mse + nodeoff_mse] / tmp_lnrows;
          T tmp_mse_right =
            mseout[coloff_mse + binoff_mse + nodeoff_mse + 1] / tmp_rnrows;

          T impurity = (tmp_lnrows * 1.0 / totalrows) * tmp_mse_left +
                       (tmp_rnrows * 1.0 / totalrows) * tmp_mse_right;
          float info_gain =
            (float)(sparsetree[parentid].best_metric_val - impurity);

          // Compute best information col_gain so far
          if (info_gain > gain[nodecnt]) {
            gain[nodecnt] = info_gain;
            best_bin_id = binid;
            best_col_id = colselector[colid];
            bestmean_left = tmp_meanleft;
            bestmean_right = tmp_meanright;
            bestcount_left = tmp_lnrows;
            bestcount_right = tmp_rnrows;
            bestmetric_left = tmp_mse_left;
            bestmetric_right = tmp_mse_right;
          }
        }
      }
      split_colidx[nodecnt] = best_col_id;
      split_binidx[nodecnt] = best_bin_id;
      //Sparse Tree
      SparseTreeNode<T, T> &curr_node = sparsetree[sparsesize + sparse_nodeid];
      curr_node.colid = split_colidx[nodecnt];
      T *dummy = nullptr;
      curr_node.quesval =
        getQuesValue(dummy, quantile, nbins, split_colidx[nodecnt],
                     split_binidx[nodecnt], split_algo);
      curr_node.left_child_id = sparsetree_sz + 2 * nodecnt;
      sparse_meanstate[curr_node.left_child_id] = bestmean_left;
      sparse_meanstate[curr_node.left_child_id + 1] = bestmean_right;
      sparse_countstate[curr_node.left_child_id] = bestcount_left;
      sparse_countstate[curr_node.left_child_id + 1] = bestcount_right;
      SparseTreeNode<T, T> leftnode, rightnode;
      leftnode.best_metric_val = bestmetric_left;
      rightnode.best_metric_val = bestmetric_right;
      sparsetree.push_back(leftnode);
      sparsetree.push_back(rightnode);
    }
    MLCommon::updateDevice(d_split_binidx, split_binidx, n_nodes,
                           tempmem->stream);
    MLCommon::updateDevice(d_split_colidx, split_colidx, n_nodes,
                           tempmem->stream);
  }
}

template <typename T>
void leaf_eval_regression(float *gain, int curr_depth, const int max_depth,
                          const int max_leaves, unsigned int *new_node_flags,
                          std::vector<SparseTreeNode<T, T>> &sparsetree,
                          const int sparsesize, std::vector<T> &sparse_mean,
                          int &n_nodes_next, std::vector<int> &sparse_nodelist,
                          int &tree_leaf_cnt) {
  std::vector<int> tmp_sparse_nodelist(sparse_nodelist);
  sparse_nodelist.clear();

  int non_leaf_counter = 0;
  bool condition_global = (curr_depth == max_depth);
  if (max_leaves != -1)
    condition_global = condition_global || (tree_leaf_cnt >= max_leaves);

  for (int i = 0; i < tmp_sparse_nodelist.size(); i++) {
    unsigned int node_flag;
    int sparse_nodeid = tmp_sparse_nodelist[i];
    T nodemean = sparse_mean[sparsesize + sparse_nodeid];
    bool condition = condition_global || (gain[i] == 0.0);
    if (condition) {
      node_flag = 0xFFFFFFFF;
      sparsetree[sparsesize + sparse_nodeid].colid = -1;
      sparsetree[sparsesize + sparse_nodeid].prediction = nodemean;
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

template <typename T>
void init_parent_value(std::vector<T> &sparse_meanstate,
                       std::vector<unsigned int> &sparse_countstate,
                       std::vector<int> &sparse_nodelist, const int sparsesize,
                       const int depth,
                       std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  T *h_predout = tempmem->h_predout->data();
  unsigned int *h_count = tempmem->h_count->data();
  int n_nodes = sparse_nodelist.size();
  for (int i = 0; i < n_nodes; i++) {
    int sparse_nodeid = sparse_nodelist[i];
    h_predout[i] = sparse_meanstate[sparsesize + sparse_nodeid];
    h_count[i] = sparse_countstate[sparsesize + sparse_nodeid];
  }
  MLCommon::updateDevice(tempmem->d_parent_pred->data(), h_predout, n_nodes,
                         tempmem->stream);
  MLCommon::updateDevice(tempmem->d_parent_count->data(), h_count, n_nodes,
                         tempmem->stream);
}
