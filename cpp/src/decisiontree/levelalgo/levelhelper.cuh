#pragma once
#include "levelkernel.cuh"

template <typename T>
void get_me_histogram(T *data, int *labels, unsigned int *flags,
                      const int nrows, const int ncols,
                      const int n_unique_labels, const int nbins,
                      const int n_nodes,
                      const std::shared_ptr<TemporaryMemory<T, int>> tempmem,
                      unsigned int *histout) {
  size_t shmem = nbins * n_unique_labels * sizeof(int) * n_nodes;
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);

  get_me_hist_kernel<<<blocks, threads, shmem, tempmem->stream>>>(
    data, labels, flags, nrows, ncols, n_unique_labels, nbins, n_nodes,
    tempmem->d_quantile->data(), histout);
}
template <typename T, typename F>
void get_me_best_split(unsigned int *hist,
                       const std::vector<unsigned int> &colselector,
                       const int nbins, const int n_unique_labels,
                       const int n_nodes, const int depth,
                       std::vector<float> &gain,
                       std::vector<std::vector<int>> &histstate,
                       std::vector<FlatTreeNode<T>> &flattree,
                       std::vector<int> &nodelist, int *split_colidx,
                       int *split_binidx, T *quantile) {
  gain.resize(pow(2, depth), 0);
  size_t n_nodes_before = 0;

  for (int i = 0; i <= (depth - 1); i++) {
    n_nodes_before += pow(2, i);
  }

  for (int i = 0; i < pow(2, depth); i++) {
    FlatTreeNode<T> leftnode, rightnode;
    std::vector<int> tmp_histleft(n_unique_labels, 0);
    std::vector<int> tmp_histright(n_unique_labels, 0);
    flattree.push_back(leftnode);
    flattree.push_back(rightnode);
    histstate.push_back(tmp_histleft);
    histstate.push_back(tmp_histright);
  }

  for (int nodecnt = 0; nodecnt < n_nodes; nodecnt++) {
    std::vector<T> bestmetric(2, 0);
    int nodeoffset = nodecnt * nbins * n_unique_labels;
    int nodeid = nodelist[nodecnt];
    int parentid = nodeid + n_nodes_before;
    int best_col_id = -1;
    int best_bin_id = -1;
    int ncols = colselector.size();
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
    //printf("Node id %d, quantile_val %f, col id %d\n", nodeid,
    //       quantile[best_col_id * nbins + best_bin_id], best_col_id);
    histstate[2 * nodeid + n_nodes_before + pow(2, depth)] = besthist_left;
    histstate[2 * nodeid + 1 + n_nodes_before + pow(2, depth)] = besthist_right;
    flattree[nodeid + n_nodes_before].colid = best_col_id;
    flattree[nodeid + n_nodes_before].quesval =
      quantile[best_col_id * nbins + best_bin_id];

    flattree[2 * nodeid + n_nodes_before + pow(2, depth)].best_metric_val =
      bestmetric[0];
    flattree[2 * nodeid + 1 + n_nodes_before + pow(2, depth)].best_metric_val =
      bestmetric[1];
  }
}

template <typename T>
void make_level_split(T *data, const int nrows, const int ncols,
                      const int nbins, const int n_nodes, int *split_colidx,
                      int *split_binidx, const unsigned int *new_node_flags,
                      unsigned int *flags,
                      const std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  split_level_kernel<<<threads, blocks, 0, tempmem->stream>>>(
    data, tempmem->d_quantile->data(), split_colidx, split_binidx, nrows, ncols,
    nbins, n_nodes, new_node_flags, flags);
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
