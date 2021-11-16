/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <test_utils.h>
#include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

#include <cuml/datasets/make_blobs.hpp>
#include <cuml/neighbors/knn.hpp>

namespace ML {

using namespace raft::random;
using namespace std;

struct KNNInputs {
  int n_rows;
  int n_cols;
  int n_centers;

  int n_query_row;

  int n_neighbors;
  int n_parts;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const KNNInputs& dims)
{
  return os;
}

template <typename T>
void gen_blobs(
  raft::handle_t& handle, T* out, int* l, int rows, int cols, int centers, const T* centroids)
{
  Datasets::make_blobs(handle,
                       out,
                       l,
                       rows,
                       cols,
                       centers,
                       true,
                       centroids,
                       nullptr,
                       0.1f,
                       true,
                       -10.0f,
                       10.0f,
                       1234ULL);
}

void create_index_parts(raft::handle_t& handle,
                        float* query_data,
                        int* query_labels,
                        vector<float*>& part_inputs,
                        vector<int*>& part_labels,
                        vector<int>& part_sizes,
                        const KNNInputs& params,
                        const float* centers)
{
  cudaStream_t stream = handle.get_stream();
  gen_blobs<float>(handle,
                   query_data,
                   query_labels,
                   params.n_rows * params.n_parts,
                   params.n_cols,
                   params.n_centers,
                   centers);

  for (int i = 0; i < params.n_parts; i++) {
    part_inputs.push_back(query_data + (i * params.n_rows * params.n_cols));
    part_labels.push_back(query_labels + (i * params.n_rows));
    part_sizes.push_back(params.n_rows);
  }
}

__global__ void to_float(float* out, int* in, int size)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= size) return;
  out[element] = float(in[element]);
}

__global__ void build_actual_output(
  int* output, int n_rows, int k, const int* idx_labels, const int64_t* indices)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  int ind         = (int)indices[element];
  output[element] = idx_labels[ind];
}

__global__ void build_expected_output(int* output, int n_rows, int k, const int* labels)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 protected:
  void testBruteForce()
  {
    cudaStream_t stream = handle.get_stream();

    raft::allocate(
      actual_labels, params.n_query_row * params.n_neighbors * params.n_parts, stream, true);
    raft::allocate(
      expected_labels, params.n_query_row * params.n_neighbors * params.n_parts, stream, true);

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data,
                    params.n_query_row,
                    output_indices,
                    output_dists,
                    params.n_neighbors,
                    true,
                    true);

    build_actual_output<<<raft::ceildiv(params.n_query_row * params.n_neighbors, 32),
                          32,
                          0,
                          stream>>>(
      actual_labels, params.n_query_row, params.n_neighbors, index_labels, output_indices);

    build_expected_output<<<raft::ceildiv(params.n_query_row, 32), 32, 0, stream>>>(
      expected_labels, params.n_query_row, params.n_neighbors, search_labels);

    ASSERT_TRUE(devArrMatch(expected_labels,
                            actual_labels,
                            params.n_query_row * params.n_neighbors,
                            raft::Compare<int>()));
  }

  void testClassification()
  {
    cudaStream_t stream = handle.get_stream();

    raft::allocate(actual_labels, params.n_query_row, stream, true);
    raft::allocate(expected_labels, params.n_query_row, stream, true);

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data,
                    params.n_query_row,
                    output_indices,
                    output_dists,
                    params.n_neighbors,
                    true,
                    true);

    vector<int*> full_labels(1);
    full_labels[0] = index_labels;

    knn_classify(handle,
                 actual_labels,
                 output_indices,
                 full_labels,
                 params.n_rows * params.n_parts,
                 params.n_query_row,
                 params.n_neighbors);

    ASSERT_TRUE(
      devArrMatch(search_labels, actual_labels, params.n_query_row, raft::Compare<int>()));
  }

  void testRegression()
  {
    cudaStream_t stream = handle.get_stream();

    raft::allocate(actual_labels, params.n_query_row, stream, true);
    raft::allocate(expected_labels, params.n_query_row, stream, true);

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data,
                    params.n_query_row,
                    output_indices,
                    output_dists,
                    params.n_neighbors,
                    true,
                    true);

    rmm::device_uvector<float> index_labels_float(params.n_rows * params.n_parts, stream);
    rmm::device_uvector<float> query_labels_float(params.n_query_row, stream);
    to_float<<<raft::ceildiv((int)index_labels_float.size(), 32), 32, 0, stream>>>(
      index_labels_float.data(), index_labels, index_labels_float.size());
    to_float<<<raft::ceildiv(params.n_query_row, 32), 32, 0, stream>>>(
      query_labels_float.data(), search_labels, params.n_query_row);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaPeekAtLastError());

    rmm::device_uvector<float> actual_labels_float(params.n_query_row, stream);

    vector<float*> full_labels(1);
    full_labels[0] = index_labels_float.data();

    knn_regress(handle,
                actual_labels_float.data(),
                output_indices,
                full_labels,
                params.n_rows,
                params.n_query_row,
                params.n_neighbors);

    ASSERT_TRUE(raft::devArrMatch(query_labels_float.data(),
                                  actual_labels_float.data(),
                                  params.n_query_row,
                                  raft::Compare<float>()));
  }

  void SetUp() override
  {
    cudaStream_t stream = handle.get_stream();

    params = ::testing::TestWithParam<KNNInputs>::GetParam();

    raft::allocate(index_data, params.n_rows * params.n_cols * params.n_parts, stream, true);
    raft::allocate(index_labels, params.n_rows * params.n_parts, stream, true);

    raft::allocate(search_data, params.n_query_row * params.n_cols, stream, true);
    raft::allocate(search_labels, params.n_query_row, stream, true);

    raft::allocate(
      output_indices, params.n_query_row * params.n_neighbors * params.n_parts, stream, true);
    raft::allocate(
      output_dists, params.n_query_row * params.n_neighbors * params.n_parts, stream, true);
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(index_data));
    CUDA_CHECK(cudaFree(index_labels));
    CUDA_CHECK(cudaFree(search_data));
    CUDA_CHECK(cudaFree(search_labels));
    CUDA_CHECK(cudaFree(output_dists));
    CUDA_CHECK(cudaFree(output_indices));
    CUDA_CHECK(cudaFree(actual_labels));
    CUDA_CHECK(cudaFree(expected_labels));
  }

 private:
  void create_data()
  {
    cudaStream_t stream = handle.get_stream();

    rmm::device_uvector<T> rand_centers(params.n_centers * params.n_cols, stream);
    Rng r(0, GeneratorType::GenPhilox);
    r.uniform(rand_centers.data(), params.n_centers * params.n_cols, -10.0f, 10.0f, stream);

    // Create index parts
    create_index_parts(handle,
                       index_data,
                       index_labels,
                       part_inputs,
                       part_labels,
                       part_sizes,
                       params,
                       rand_centers.data());

    gen_blobs(handle,
              search_data,
              search_labels,
              params.n_query_row,
              params.n_cols,
              params.n_centers,
              rand_centers.data());
  }

  raft::handle_t handle;

  KNNInputs params;

  float* index_data;
  int* index_labels;

  vector<float*> part_inputs;
  vector<int*> part_labels;
  vector<int> part_sizes;

  float* search_data;
  int* search_labels;

  float* output_dists;
  int64_t* output_indices;

  int* actual_labels;
  int* expected_labels;
};

const std::vector<KNNInputs> inputs = {{50, 5, 2, 25, 5, 2},
                                       {50, 5, 2, 25, 10, 2},
                                       {500, 5, 2, 25, 5, 7},
                                       {500, 50, 2, 25, 10, 7},
                                       {500, 50, 7, 25, 5, 7},
                                       {50, 5, 3, 15, 5, 7}};

typedef KNNTest<float> KNNTestF;
TEST_P(KNNTestF, BruteForce) { this->testBruteForce(); }
TEST_P(KNNTestF, Classification) { this->testClassification(); }
TEST_P(KNNTestF, Regression) { this->testRegression(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestF, ::testing::ValuesIn(inputs));

}  // end namespace ML
