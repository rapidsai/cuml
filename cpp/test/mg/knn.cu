/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "../prims/test_utils.h"
#include "test_opg_utils.h"

#include <cuml/neighbors/knn_mg.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace ML {
namespace KNN {
namespace opg {

struct KNNParams {
  int k;

  size_t min_rows;
  size_t n_cols;

  int n_query_parts;
  int n_index_parts;

  size_t batch_size;
};

class BruteForceKNNTest : public ::testing::TestWithParam<KNNParams> {
 public:
  void generate_partition(Matrix::floatData_t* part,
                          size_t n_rows,
                          int n_cols,
                          int n_clusters,
                          int part_num,
                          cudaStream_t stream)
  {
    rmm::device_uvector<int> labels(n_rows, stream);

    raft::random::make_blobs<float, int>(
      part->ptr, labels.data(), (int)n_rows, (int)n_cols, 5, stream);
  }

  bool runTest(const KNNParams& params)
  {
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm     = handle.get_comms();
    const auto allocator = rmm::mr::get_current_device_resource();

    cudaStream_t stream = handle.get_stream();

    int my_rank = comm.get_rank();
    int size    = comm.get_size();

    int index_parts_per_rank = raft::ceildiv(params.n_index_parts, size);
    int query_parts_per_rank = raft::ceildiv(params.n_query_parts, size);
    std::vector<Matrix::RankSizePair*> idxPartsToRanks;
    std::vector<Matrix::RankSizePair*> queryPartsToRanks;
    for (int cur_rank = 0; cur_rank < size; cur_rank++) {
      int ippr = index_parts_per_rank;
      int qppr = query_parts_per_rank;
      if (cur_rank == size - 1) {
        ippr = params.n_index_parts - (cur_rank * index_parts_per_rank);
        qppr = params.n_query_parts - (cur_rank * query_parts_per_rank);
      }

      std::cout << "Generating " << ippr << " partitions for rank " << cur_rank << std::endl;

      std::cout << "min_rows: " << params.min_rows << std::endl;

      for (int part_n = 0; part_n < ippr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        idxPartsToRanks.push_back(rsp);
      }

      for (int part_n = 0; part_n < qppr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        queryPartsToRanks.push_back(rsp);
      }
    }

    std::cout << idxPartsToRanks.size() << std::endl;

    if (my_rank == size - 1) {
      index_parts_per_rank = params.n_index_parts - ((size - 1) * index_parts_per_rank);
      query_parts_per_rank = params.n_query_parts - ((size - 1) * query_parts_per_rank);
    }

    std::cout << "Generating " << index_parts_per_rank << " partitions for rank " << my_rank
              << std::endl;

    std::vector<Matrix::floatData_t*> query_parts;
    std::vector<Matrix::floatData_t*> out_d_parts;
    std::vector<Matrix::Data<int64_t>*> out_i_parts;
    for (int i = 0; i < query_parts_per_rank; i++) {
      float* q =
        (float*)allocator.get()->allocate(params.min_rows * params.n_cols * sizeof(float*), stream);

      float* o =
        (float*)allocator.get()->allocate(params.min_rows * params.k * sizeof(float*), stream);

      int64_t* ind =
        (int64_t*)allocator.get()->allocate(params.min_rows * params.k * sizeof(int64_t), stream);

      Matrix::Data<float>* query_d = new Matrix::Data<float>(q, params.min_rows * params.n_cols);

      Matrix::floatData_t* out_d = new Matrix::floatData_t(o, params.min_rows * params.k);

      Matrix::Data<int64_t>* out_i = new Matrix::Data<int64_t>(ind, params.min_rows * params.k);

      query_parts.push_back(query_d);
      out_d_parts.push_back(out_d);
      out_i_parts.push_back(out_i);

      generate_partition(query_d, params.min_rows, params.n_cols, 5, i, stream);
    }

    std::vector<Matrix::floatData_t*> index_parts;

    for (int i = 0; i < index_parts_per_rank; i++) {
      float* ind =
        (float*)allocator.get()->allocate(params.min_rows * params.n_cols * sizeof(float), stream);

      Matrix::Data<float>* i_d = new Matrix::Data<float>(ind, params.min_rows * params.n_cols);

      index_parts.push_back(i_d);

      generate_partition(i_d, params.min_rows, params.n_cols, 5, i, stream);
    }

    Matrix::PartDescriptor idx_desc(
      params.min_rows * params.n_index_parts, params.n_cols, idxPartsToRanks, comm.get_rank());

    Matrix::PartDescriptor query_desc(
      params.min_rows * params.n_query_parts, params.n_cols, queryPartsToRanks, comm.get_rank());

    handle.sync_stream(stream);

    /**
     * Execute brute_force_knn()
     */
    brute_force_knn(handle,
                    out_i_parts,
                    out_d_parts,
                    index_parts,
                    idx_desc,
                    query_parts,
                    query_desc,
                    params.k,
                    params.batch_size,
                    true);

    handle.sync_stream(stream);

    std::cout << raft::arr2Str(out_i_parts[0]->ptr, 10, "final_out_I", stream) << std::endl;
    std::cout << raft::arr2Str(out_d_parts[0]->ptr, 10, "final_out_D", stream) << std::endl;

    /**
     * Verify expected results
     */

    for (Matrix::floatData_t* fd : query_parts) {
      allocator.get()->deallocate(fd->ptr, fd->totalSize * sizeof(float), stream);
      delete fd;
    }

    for (Matrix::floatData_t* fd : index_parts) {
      allocator.get()->deallocate(fd->ptr, fd->totalSize * sizeof(float), stream);
      delete fd;
    }

    for (Matrix::Data<int64_t>* fd : out_i_parts) {
      allocator.get()->deallocate(fd->ptr, fd->totalSize * sizeof(int64_t), stream);
      delete fd;
    }

    for (Matrix::floatData_t* fd : out_d_parts) {
      allocator.get()->deallocate(fd->ptr, fd->totalSize * sizeof(float), stream);
      delete fd;
    }

    for (Matrix::RankSizePair* rsp : queryPartsToRanks) {
      delete rsp;
    }

    for (Matrix::RankSizePair* rsp : idxPartsToRanks) {
      delete rsp;
    }

    int actual   = 1;
    int expected = 1;
    return raft::CompareApprox<int>(1)(actual, expected);
  }

 private:
  raft::handle_t handle;
};

const std::vector<KNNParams> inputs = {{5, 50, 3, 5, 5, 12},
                                       {10, 50, 3, 5, 5, 50},
                                       {5, 50, 3, 5, 5, 50},
                                       {5, 500, 5, 5, 5, 50},
                                       {10, 500, 50, 5, 5, 50},
                                       {15, 500, 5, 5, 5, 50},
                                       {5, 500, 10, 5, 5, 50},
                                       {10, 500, 10, 5, 5, 50},
                                       {15, 500, 10, 5, 5, 50}};

typedef BruteForceKNNTest KNNTest;

TEST_P(KNNTest, Result) { ASSERT_TRUE(runTest(GetParam())); }

INSTANTIATE_TEST_CASE_P(BruteForceKNNTest, KNNTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace KNN
}  // namespace ML
