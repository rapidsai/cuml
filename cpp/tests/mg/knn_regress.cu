/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "knn_test_helper.cuh"

namespace ML {
namespace KNN {
namespace opg {

template <>
void generate_partitions(float* data,
                         float* outputs,
                         size_t n_rows,
                         int n_cols,
                         int n_clusters,
                         int my_rank,
                         cudaStream_t stream)
{
  Random::make_blobs<float, int>(data,
                                 (int*)outputs,
                                 (int)n_rows,
                                 (int)n_cols,
                                 n_clusters,
                                 stream,
                                 true,
                                 nullptr,
                                 nullptr,
                                 1.0,
                                 -10.0,
                                 10.0,
                                 my_rank);
  raft::linalg::convert_array(outputs, (int*)outputs, n_rows, stream);
}

class KNNRegressTest : public ::testing::TestWithParam<KNNParams> {
 public:
  bool runTest(const KNNParams& params)
  {
    KNNTestHelper<float> knn_th;
    knn_th.generate_data(params);

    /**
     * Execute knn_regress()
     */
    knn_regress(knn_th.handle,
                &(knn_th.out_parts),
                &(knn_th.out_i_parts),
                &(knn_th.out_d_parts),
                knn_th.index_parts,
                *(knn_th.idx_desc),
                knn_th.query_parts,
                *(knn_th.query_desc),
                knn_th.y,
                false,
                false,
                params.k,
                params.n_outputs,
                params.batch_size,
                true);

    knn_th.display_results();
    knn_th.release_ressources(params);

    int actual   = 1;
    int expected = 1;
    return raft::CompareApprox<int>(1)(actual, expected);
  }
};

const std::vector<KNNParams> inputs = {{5, 1, 8, 50, 3, 2, 2, 12}};

typedef KNNRegressTest KNNReTest;

TEST_P(KNNReTest, Result) { ASSERT_TRUE(runTest(GetParam())); }

INSTANTIATE_TEST_CASE_P(KNNRegressTest, KNNReTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace KNN
}  // namespace ML
