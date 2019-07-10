#pragma once
#include "levelkernel.cuh"

template <typename T>
void get_me_histogram(T *data, int *labels, unsigned int *flags,
                      const int nrows, const int ncols,
                      const int n_unique_labels, const int nbins,
                      const int n_nodes, const int maxnodes,
                      LevelTemporaryMemory<T> *tempmem, unsigned int *histout) {
  size_t histcount = ncols * nbins * n_unique_labels * n_nodes;
  CUDA_CHECK(cudaMemsetAsync(histout, 0, histcount * sizeof(unsigned int),
                             tempmem->stream));
  int node_batch = min(n_nodes, maxnodes);
  size_t shmem = nbins * n_unique_labels * sizeof(int) * node_batch;
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  if ((n_nodes == node_batch) && (blocks < 65536)) {
    get_me_hist_kernel<<<blocks, threads, shmem, tempmem->stream>>>(
      data, labels, flags, nrows, ncols, n_unique_labels, nbins, n_nodes,
      tempmem->d_quantile->data(), histout);
  } else {
    /*get_me_hist_kernel_batched<<<blocks, threads, shmem, tempmem->stream>>>(
      data, labels, flags, nrows, ncols, n_unique_labels, nbins, n_nodes,
      tempmem->d_quantile->data(), node_batch, histout);*/
    get_me_hist_kernel_global<<<blocks, threads, 0, tempmem->stream>>>(
      data, labels, flags, nrows, ncols, n_unique_labels, nbins, n_nodes,
      tempmem->d_quantile->data(), histout);
  }
  CUDA_CHECK(cudaGetLastError());
}
template <typename T, typename F, typename DF>
void get_me_best_split(
  unsigned int *hist, unsigned int *d_hist,
  const std::vector<unsigned int> &colselector, const int nbins,
  const int n_unique_labels, const int n_nodes, const int depth,
  std::vector<float> &gain, std::vector<std::vector<int>> &histstate,
  std::vector<FlatTreeNode<T>> &flattree, std::vector<int> &nodelist,
  int *split_colidx, int *split_binidx, int *d_split_colidx,
  int *d_split_binidx, LevelTemporaryMemory<T> *leveltempmem) {
  T *quantile = leveltempmem->h_quantile->data();
  int ncols = colselector.size();
  size_t histcount = ncols * nbins * n_unique_labels * n_nodes;
  bool use_gpu_flag = false;
  if (depth > 6) use_gpu_flag = true;
  gain.resize(pow(2, depth), 0);
  size_t n_nodes_before = 0;
  for (int i = 0; i <= (depth - 1); i++) {
    n_nodes_before += pow(2, i);
  }
  if (use_gpu_flag) {
    //GPU based best split
    unsigned int *h_parent_hist, *d_parent_hist, *d_child_hist, *h_child_hist;
    T *d_parent_metric, *d_child_best_metric;
    T *h_parent_metric, *h_child_best_metric;
    float *d_outgain, *h_outgain;
    h_parent_hist = leveltempmem->h_parent_hist->data();
    h_child_hist = leveltempmem->h_child_hist->data();
    h_parent_metric = leveltempmem->h_parent_metric->data();
    h_child_best_metric = leveltempmem->h_child_best_metric->data();
    h_outgain = leveltempmem->h_outgain->data();

    d_parent_hist = leveltempmem->d_parent_hist->data();
    d_child_hist = leveltempmem->d_child_hist->data();
    d_parent_metric = leveltempmem->d_parent_metric->data();
    d_child_best_metric = leveltempmem->d_child_best_metric->data();
    d_outgain = leveltempmem->d_outgain->data();
    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int nodeid = nodelist[nodecnt];
      int parentid = nodeid + n_nodes_before;
      std::vector<int> &parent_hist = histstate[parentid];
      h_parent_metric[nodecnt] = flattree[parentid].best_metric_val;
      for (int j = 0; j < n_unique_labels; j++) {
        h_parent_hist[nodecnt * n_unique_labels + j] = parent_hist[j];
      }
    }

    MLCommon::updateDevice(d_parent_hist, h_parent_hist,
                           n_nodes * n_unique_labels, leveltempmem->stream);
    MLCommon::updateDevice(d_parent_metric, h_parent_metric, n_nodes,
                           leveltempmem->stream);
    int threads = 64;
    size_t shmemsz = (threads + 2) * 2 * n_unique_labels * sizeof(int);
    get_me_best_split_kernel<T, DF>
      <<<n_nodes, threads, shmemsz, leveltempmem->stream>>>(
        d_hist, d_parent_hist, d_parent_metric, nbins, ncols, n_nodes,
        n_unique_labels, d_outgain, d_split_colidx, d_split_binidx,
        d_child_hist, d_child_best_metric);
    CUDA_CHECK(cudaGetLastError());
    MLCommon::updateHost(h_child_hist, d_child_hist,
                         2 * n_nodes * n_unique_labels, leveltempmem->stream);
    MLCommon::updateHost(h_outgain, d_outgain, n_nodes, leveltempmem->stream);
    MLCommon::updateHost(h_child_best_metric, d_child_best_metric, 2 * n_nodes,
                         leveltempmem->stream);
    MLCommon::updateHost(split_binidx, d_split_binidx, n_nodes,
                         leveltempmem->stream);
    MLCommon::updateHost(split_colidx, d_split_colidx, n_nodes,
                         leveltempmem->stream);

    CUDA_CHECK(cudaStreamSynchronize(leveltempmem->stream));
    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      int nodeid = nodelist[nodecnt];
      gain[nodeid] = h_outgain[nodecnt];
      for (int j = 0; j < n_unique_labels; j++) {
        histstate[2 * nodeid + n_nodes_before + pow(2, depth)][j] =
          h_child_hist[n_unique_labels * nodecnt * 2 + j];
        histstate[2 * nodeid + 1 + n_nodes_before + pow(2, depth)][j] =
          h_child_hist[n_unique_labels * nodecnt * 2 + j + n_unique_labels];
      }
      flattree[nodeid + n_nodes_before].colid = split_colidx[nodecnt];
      flattree[nodeid + n_nodes_before].quesval =
        quantile[split_colidx[nodecnt] * nbins + split_binidx[nodecnt]];
      flattree[2 * nodeid + n_nodes_before + pow(2, depth)].best_metric_val =
        h_child_best_metric[2 * nodecnt];
      flattree[2 * nodeid + 1 + n_nodes_before + pow(2, depth)]
        .best_metric_val = h_child_best_metric[2 * nodecnt + 1];
    }
  } else {
    MLCommon::updateHost(hist, d_hist, histcount, leveltempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(leveltempmem->stream));

    for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
      std::vector<T> bestmetric(2, 0);
      int nodeoffset = nodecnt * nbins * n_unique_labels;
      int nodeid = nodelist[nodecnt];
      int parentid = nodeid + n_nodes_before;
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
          std::vector<int> tmp_histleft(n_unique_labels);
          std::vector<int> tmp_histright(n_unique_labels);

          std::vector<int> &parent_hist = histstate[parentid];
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
          if (tmp_lnrows == 0 || tmp_rnrows == 0) continue;

          float tmp_gini_left = F::exec(tmp_histleft, tmp_lnrows);
          float tmp_gini_right = F::exec(tmp_histright, tmp_rnrows);

          ASSERT((tmp_gini_left >= 0.0f) && (tmp_gini_left <= 1.0f),
                 "gini left value %f not in [0.0, 1.0]", tmp_gini_left);
          ASSERT((tmp_gini_right >= 0.0f) && (tmp_gini_right <= 1.0f),
                 "gini right value %f not in [0.0, 1.0]", tmp_gini_right);

          float impurity = (tmp_lnrows * 1.0f / totalrows) * tmp_gini_left +
                           (tmp_rnrows * 1.0f / totalrows) * tmp_gini_right;
          float info_gain = flattree[parentid].best_metric_val - impurity;

          // Compute best information col_gain so far
          if (info_gain > gain[nodeid]) {
            gain[nodeid] = info_gain;
            best_bin_id = binid;
            best_col_id = colid;
            besthist_left = tmp_histleft;
            besthist_right = tmp_histright;
            bestmetric[0] = tmp_gini_left;
            bestmetric[1] = tmp_gini_right;
          }
        }
      }
      split_colidx[nodecnt] = best_col_id;
      split_binidx[nodecnt] = best_bin_id;
      histstate[2 * nodeid + n_nodes_before + pow(2, depth)] = besthist_left;
      histstate[2 * nodeid + 1 + n_nodes_before + pow(2, depth)] =
        besthist_right;
      flattree[nodeid + n_nodes_before].colid = best_col_id;
      flattree[nodeid + n_nodes_before].quesval =
        quantile[best_col_id * nbins + best_bin_id];

      flattree[2 * nodeid + n_nodes_before + pow(2, depth)].best_metric_val =
        bestmetric[0];
      flattree[2 * nodeid + 1 + n_nodes_before + pow(2, depth)]
        .best_metric_val = bestmetric[1];
    }
    MLCommon::updateDevice(d_split_binidx, split_binidx, n_nodes,
                           leveltempmem->stream);
    MLCommon::updateDevice(d_split_colidx, split_colidx, n_nodes,
                           leveltempmem->stream);
  }
}

template <typename T>
void make_level_split(T *data, const int nrows, const int ncols,
                      const int nbins, const int n_nodes, int *split_colidx,
                      int *split_binidx, const unsigned int *new_node_flags,
                      unsigned int *flags, LevelTemporaryMemory<T> *tempmem) {
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  split_level_kernel<<<blocks, threads, 0, tempmem->stream>>>(
    data, tempmem->d_quantile->data(), split_colidx, split_binidx, nrows, ncols,
    nbins, n_nodes, new_node_flags, flags);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
ML::DecisionTree::TreeNode<T, int> *go_recursive(
  std::vector<FlatTreeNode<T>> &flattree, int idx = 0) {
  ML::DecisionTree::TreeNode<T, int> *node = NULL;
  if (idx < flattree.size()) {
    node = new ML::DecisionTree::TreeNode<T, int>();
    node->split_metric_val = flattree[idx].best_metric_val;
    node->question.column = flattree[idx].colid;
    node->question.value = flattree[idx].quesval;
    node->prediction = flattree[idx].prediction;
    if (flattree[idx].type == true) {
      return node;
    }
    node->left = go_recursive(flattree, 2 * idx + 1);
    node->right = go_recursive(flattree, 2 * idx + 2);
  }
  return node;
}

template <typename T>
void leaf_eval(std::vector<float> &gain, int curr_depth, int max_depth,
               unsigned int *new_node_flags,
               std::vector<FlatTreeNode<T>> &flattree,
               std::vector<std::vector<int>> hist, int &n_nodes_next,
               std::vector<int> &nodelist) {
  std::vector<int> tmp_nodelist(nodelist);
  nodelist.clear();
  int n_nodes_before = 0;
  for (int i = 0; i <= (curr_depth - 1); i++) {
    n_nodes_before += pow(2, i);
  }
  int leaf_counter = 0;
  for (int i = 0; i < tmp_nodelist.size(); i++) {
    unsigned int node_flag;
    int nodeid = tmp_nodelist[i];
    if (gain[nodeid] == 0.0 || curr_depth == max_depth) {
      node_flag = 0xFFFFFFFF;
      flattree[n_nodes_before + nodeid].type = true;
      flattree[n_nodes_before + nodeid].prediction =
        get_class_hist(hist[n_nodes_before + nodeid]);
    } else {
      nodelist.push_back(2 * nodeid);
      nodelist.push_back(2 * nodeid + 1);
      node_flag = leaf_counter;
      leaf_counter++;
    }
    new_node_flags[i] = node_flag;
  }
  int nleafed = tmp_nodelist.size() - leaf_counter;
  n_nodes_next = 2 * leaf_counter;
}
