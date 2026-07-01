/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_test_helper.cuh"

namespace ML {
namespace KNN {
namespace opg {

template <>
void generate_partitions(float* data,
                         int* lbls_ptr,
                         size_t n_rows,
                         int n_cols,
                         int n_clusters,
                         int my_rank,
                         cudaStream_t stream)
{
  raft::random::make_blobs<float, int>(data,
                                       lbls_ptr,
                                       (int)n_rows,
                                       (int)n_cols,
                                       n_clusters,
                                       stream,
                                       true,
                                       nullptr,
                                       nullptr,
                                       1.0f,
                                       true,
                                       -10.0f,
                                       10.0f,
                                       static_cast<uint64_t>(my_rank));
}

class KNNClassifyTest : public ::testing::TestWithParam<KNNParams> {
 public:
  bool runTest(const KNNParams& params)
  {
    KNNTestHelper<int> knn_th;
    knn_th.generate_data(params);

    std::vector<int> n_unique;
    for (int i = 0; i < params.n_outputs; i++) {
      n_unique.push_back(params.n_classes);
    }

    std::vector<int*> uniq_labels(params.n_outputs);
    std::vector<raft::device_vector<int, int64_t>> uniq_label_storage;
    uniq_label_storage.reserve(params.n_outputs);
    for (int i = 0; i < params.n_outputs; i++) {
      int nu = n_unique[i];
      std::vector<int> ul_h(nu);
      for (int j = 0; j < nu; j++) {
        ul_h[j] = j;
      }
      uniq_label_storage.emplace_back(
        raft::make_device_vector<int, int64_t>(knn_th.handle, static_cast<int64_t>(nu)));
      uniq_labels[i] = uniq_label_storage.back().data_handle();
      raft::update_device(uniq_labels[i], ul_h.data(), ul_h.size(), knn_th.stream);
    }

    /**
     * Execute knn_classify()
     */
    knn_classify(knn_th.handle,
                 &(knn_th.out_parts),
                 nullptr,
                 knn_th.index_parts,
                 *(knn_th.idx_desc),
                 knn_th.query_parts,
                 *(knn_th.query_desc),
                 knn_th.y,
                 uniq_labels,
                 n_unique,
                 false,
                 false,
                 false,
                 params.k,
                 params.batch_size,
                 true);

    knn_th.display_results();
    knn_th.release_ressources(params);

    int actual   = 1;
    int expected = 1;
    return MLCommon::CompareApprox<int>(1)(actual, expected);
  }
};

const std::vector<KNNParams> inputs = {{5, 1, 8, 50, 3, 2, 2, 12}};

typedef KNNClassifyTest KNNClTest;

TEST_P(KNNClTest, Result) { ASSERT_TRUE(runTest(GetParam())); }

INSTANTIATE_TEST_CASE_P(KNNClassifyTest, KNNClTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace KNN
}  // namespace ML
