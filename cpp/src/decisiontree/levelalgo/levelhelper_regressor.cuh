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
void initial_metric_regression(T *labels, unsigned int *sample_cnt,
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
  if (blocks > 65536) blocks = 65536;
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
void get_mse_regression(T *data, T *labels, unsigned int *flags,
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
  if ((n_nodes == node_batch_pred) && (blocks < 65536)) {
    get_pred_kernel<<<blocks, threads, shmempred, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(), d_predout, d_count);
  } else {
    get_pred_kernel_global<<<blocks, threads, 0, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(), d_predout, d_count);
  }
  CUDA_CHECK(cudaGetLastError());
  if ((n_nodes == node_batch_mse) && (blocks < 65536)) {
    get_mse_kernel<T, F><<<blocks, threads, shmemmse, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(),
      tempmem->d_parent_pred->data(), tempmem->d_parent_count->data(),
      d_predout, d_count, d_mseout);
  } else {
    get_mse_kernel_global<T, F><<<blocks, threads, 0, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(),
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
  const int min_rpn, std::vector<float> &gain, std::vector<T> &meanstate,
  std::vector<unsigned int> &countstate,
  std::vector<FlatTreeNode<T, T>> &flattree, std::vector<int> &nodelist,
  int *split_colidx, int *split_binidx, int *d_split_colidx,
  int *d_split_binidx, std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  T *quantile = tempmem->h_quantile->data();
  int ncols = colselector.size();
  size_t predcount = ncols * nbins * n_nodes;
  bool use_gpu_flag = false;
  if (n_nodes > 512) use_gpu_flag = false;
  gain.resize(pow(2, depth), 0.0);
  size_t n_nodes_before = 0;
  for (int i = 0; i <= (depth - 1); i++) {
    n_nodes_before += pow(2, i);
  }
  if (use_gpu_flag) {
  } else {
    MLCommon::updateHost(mseout, d_mseout, 2 * predcount, tempmem->stream);
    MLCommon::updateHost(predout, d_predout, predcount, tempmem->stream);
    MLCommon::updateHost(count, d_count, predcount, tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      std::vector<T> bestmetric(2, 0);
      int nodeoff_mse = nodecnt * nbins * 2;
      int nodeoff_pred = nodecnt * nbins;
      int nodeid = nodelist[nodecnt];
      int parentid = nodeid + n_nodes_before;
      int best_col_id = -1;
      int best_bin_id = -1;
      T bestmean_left, bestmean_right;
      unsigned int bestcount_left, bestcount_right;
      for (int colid = 0; colid < ncols; colid++) {
        int coloff_mse = colid * nbins * 2 * n_nodes;
        int coloff_pred = colid * nbins * n_nodes;
        for (int binid = 0; binid < nbins; binid++) {
          int binoff_mse = binid * 2;
          int binoff_pred = binid;
          unsigned int tmp_lnrows = 0;
          unsigned int tmp_rnrows = 0;

          T parent_mean = meanstate[parentid];
          unsigned int parent_count = countstate[parentid];
          tmp_lnrows = count[coloff_pred + binoff_pred + nodeoff_pred];
          tmp_rnrows = parent_count - tmp_lnrows;
          unsigned int totalrows = tmp_lnrows + tmp_rnrows;
          if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows <= min_rpn)
            continue;

          T tmp_meanleft = predout[coloff_pred + binoff_pred + nodeoff_pred];
          T tmp_meanright = parent_mean * parent_count - tmp_meanleft;
          tmp_meanleft /= tmp_lnrows;
          tmp_meanright /= tmp_rnrows;
          T tmp_mse_left =
            mseout[coloff_mse + binoff_mse + nodeoff_mse] / tmp_lnrows;
          T tmp_mse_right =
            mseout[coloff_mse + binoff_mse + nodeoff_mse + 1] / tmp_lnrows;

          T impurity = (tmp_lnrows * 1.0 / totalrows) * tmp_mse_left +
                       (tmp_rnrows * 1.0 / totalrows) * tmp_mse_right;
          float info_gain =
            (float)(flattree[parentid].best_metric_val - impurity);

          // Compute best information col_gain so far
          if (info_gain > gain[nodeid]) {
            gain[nodeid] = info_gain;
            best_bin_id = binid;
            best_col_id = colselector[colid];
            bestmean_left = tmp_meanleft;
            bestmean_right = tmp_meanright;
            bestcount_left = tmp_lnrows;
            bestcount_right = tmp_rnrows;
            bestmetric[0] = tmp_mse_left;
            bestmetric[1] = tmp_mse_right;
          }
        }
      }
      split_colidx[nodecnt] = best_col_id;
      split_binidx[nodecnt] = best_bin_id;
      meanstate[2 * nodeid + n_nodes_before + pow(2, depth)] = bestmean_left;
      meanstate[2 * nodeid + 1 + n_nodes_before + pow(2, depth)] =
        bestmean_right;
      countstate[2 * nodeid + n_nodes_before + pow(2, depth)] = bestcount_left;
      countstate[2 * nodeid + 1 + n_nodes_before + pow(2, depth)] =
        bestcount_right;
      flattree[nodeid + n_nodes_before].colid = best_col_id;
      flattree[nodeid + n_nodes_before].quesval =
        quantile[best_col_id * nbins + best_bin_id];
      flattree[2 * nodeid + n_nodes_before + pow(2, depth)].best_metric_val =
        bestmetric[0];
      flattree[2 * nodeid + 1 + n_nodes_before + pow(2, depth)]
        .best_metric_val = bestmetric[1];
    }
    MLCommon::updateDevice(d_split_binidx, split_binidx, n_nodes,
                           tempmem->stream);
    MLCommon::updateDevice(d_split_colidx, split_colidx, n_nodes,
                           tempmem->stream);
  }
}

template <typename T>
void leaf_eval_regression(std::vector<float> &gain, int curr_depth,
                          const int max_depth, const int max_leaves,
                          unsigned int *new_node_flags,
                          std::vector<FlatTreeNode<T, T>> &flattree,
                          std::vector<T> &mean, int &n_nodes_next,
                          std::vector<int> &nodelist, int &tree_leaf_cnt) {
  std::vector<int> tmp_nodelist(nodelist);
  nodelist.clear();
  int n_nodes_before = 0;
  for (int i = 0; i <= (curr_depth - 1); i++) {
    n_nodes_before += pow(2, i);
  }
  int non_leaf_counter = 0;
  for (int i = 0; i < tmp_nodelist.size(); i++) {
    unsigned int node_flag;
    int nodeid = tmp_nodelist[i];
    T nodemean = mean[n_nodes_before + nodeid];
    bool condition = (gain[nodeid] == 0.0);
    condition = condition || (curr_depth == max_depth);
    if (max_leaves != -1)
      condition = condition || (tree_leaf_cnt >= max_leaves);
    if (condition) {
      node_flag = 0xFFFFFFFF;
      flattree[n_nodes_before + nodeid].type = true;
      flattree[n_nodes_before + nodeid].prediction = nodemean;
    } else {
      nodelist.push_back(2 * nodeid);
      nodelist.push_back(2 * nodeid + 1);
      node_flag = non_leaf_counter;
      non_leaf_counter++;
    }
    new_node_flags[i] = node_flag;
  }
  int nleafed = tmp_nodelist.size() - non_leaf_counter;
  tree_leaf_cnt += nleafed;
  n_nodes_next = 2 * non_leaf_counter;
}

template <typename T>
void init_parent_value(std::vector<T> &meanstate,
                       std::vector<unsigned int> &countstate,
                       std::vector<int> &nodelist,
                       std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  T *h_predout = tempmem->h_predout->data();
  unsigned int *h_count = tempmem->h_count->data();
  int n_nodes = nodelist.size();
  for (int i = 0; i < n_nodes; i++) {
    int nodeid = nodelist[i];
    h_predout[i] = meanstate[nodeid];
    h_count[i] = countstate[nodeid];
  }
  MLCommon::updateDevice(tempmem->d_parent_pred->data(), h_predout, n_nodes,
                         tempmem->stream);
  MLCommon::updateDevice(tempmem->d_parent_count->data(), h_count, n_nodes,
                         tempmem->stream);
}
