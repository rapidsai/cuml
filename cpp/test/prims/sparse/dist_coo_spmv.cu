/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/linalg/unary_op.cuh>
#include <raft/mr/device/allocator.hpp>

#include <sparse/convert/coo.cuh>
#include <sparse/distance/coo_spmv.cuh>
#include <sparse/distance/operators.cuh>

#include <test_utils.h>

namespace raft {
namespace sparse {
namespace distance {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct SparseDistanceCOOSPMVInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;

  raft::distance::DistanceType metric;

  float metric_arg = 0.0;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(
  ::std::ostream &os,
  const SparseDistanceCOOSPMVInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class SparseDistanceCOOSPMVTest
  : public ::testing::TestWithParam<
      SparseDistanceCOOSPMVInputs<value_idx, value_t>> {
 public:
  template <typename reduce_f, typename accum_f, typename write_f>
  void compute_dist(reduce_f reduce_func, accum_f accum_func,
                    write_f write_func, bool rev = true) {
    raft::mr::device::buffer<value_idx> coo_rows(
      dist_config.allocator, dist_config.stream,
      max(dist_config.b_nnz, dist_config.a_nnz));

    raft::sparse::convert::csr_to_coo(dist_config.b_indptr, dist_config.b_nrows,
                                      coo_rows.data(), dist_config.b_nnz,
                                      dist_config.stream);

    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_dists, dist_config, coo_rows.data(), reduce_func, accum_func,
      write_func);

    if (rev) {
      raft::sparse::convert::csr_to_coo(dist_config.a_indptr,
                                        dist_config.a_nrows, coo_rows.data(),
                                        dist_config.a_nnz, dist_config.stream);

      balanced_coo_pairwise_generalized_spmv_rev<value_idx, value_t>(
        out_dists, dist_config, coo_rows.data(), reduce_func, accum_func,
        write_func);
    }
  }

  void run_spmv() {
    switch (params.metric) {
      case raft::distance::DistanceType::InnerProduct:
        compute_dist(Product(), Sum(), AtomicAdd(), true);
        break;
      case raft::distance::DistanceType::L2Unexpanded:
        compute_dist(SqDiff(), Sum(), AtomicAdd());
        break;
      case raft::distance::DistanceType::Canberra:
        compute_dist(
          [] __device__(value_t a, value_t b) {
            return fabsf(a - b) / (fabsf(a) + fabsf(b));
          },
          Sum(), AtomicAdd());
        break;
      case raft::distance::DistanceType::L1:
        compute_dist(AbsDiff(), Sum(), AtomicAdd());
        break;
      case raft::distance::DistanceType::Linf:
        compute_dist(AbsDiff(), Max(), AtomicMax());
        break;
      case raft::distance::DistanceType::LpUnexpanded: {
        compute_dist(PDiff(params.metric_arg), Sum(), AtomicAdd());
        float p = 1.0f / params.metric_arg;
        raft::linalg::unaryOp<value_t>(
          out_dists, out_dists, dist_config.a_nrows * dist_config.b_nrows,
          [=] __device__(value_t input) { return powf(input, p); },
          dist_config.stream);

      } break;
      default:
        throw raft::exception("Unknown distance");
    }
  }

 protected:
  void make_data() {
    std::vector<value_idx> indptr_h = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h = params.data_h;

    allocate(indptr, indptr_h.size());
    allocate(indices, indices_h.size());
    allocate(data, data_h.size());

    update_device(indptr, indptr_h.data(), indptr_h.size(), stream);
    update_device(indices, indices_h.data(), indices_h.size(), stream);
    update_device(data, data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h = params.out_dists_ref_h;

    allocate(out_dists_ref, (indptr_h.size() - 1) * (indptr_h.size() - 1));

    update_device(out_dists_ref, out_dists_ref_h.data(), out_dists_ref_h.size(),
                  stream);
  }

  void SetUp() override {
    params = ::testing::TestWithParam<
      SparseDistanceCOOSPMVInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<raft::mr::device::allocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    make_data();

    dist_config.b_nrows = params.indptr_h.size() - 1;
    dist_config.b_ncols = params.n_cols;
    dist_config.b_nnz = params.indices_h.size();
    dist_config.b_indptr = indptr;
    dist_config.b_indices = indices;
    dist_config.b_data = data;
    dist_config.a_nrows = params.indptr_h.size() - 1;
    dist_config.a_ncols = params.n_cols;
    dist_config.a_nnz = params.indices_h.size();
    dist_config.a_indptr = indptr;
    dist_config.a_indices = indices;
    dist_config.a_data = data;
    dist_config.handle = cusparseHandle;
    dist_config.allocator = alloc;
    dist_config.stream = stream;

    int out_size = dist_config.a_nrows * dist_config.b_nrows;

    allocate(out_dists, out_size);

    ML::Logger::get().setLevel(CUML_LEVEL_DEBUG);

    run_spmv();

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(indptr));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out_dists));
    CUDA_CHECK(cudaFree(out_dists_ref));
  }

  void compare() {
    raft::print_device_vector("expected: ", out_dists_ref,
                              params.out_dists_ref_h.size(), std::cout);
    raft::print_device_vector("out_dists: ", out_dists,
                              params.out_dists_ref_h.size(), std::cout);
    ASSERT_TRUE(devArrMatch(out_dists_ref, out_dists,
                            params.out_dists_ref_h.size(),
                            CompareApprox<value_t>(1e-3)));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t cusparseHandle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_t *out_dists, *out_dists_ref;

  raft::sparse::distance::distances_config_t<value_idx, value_t> dist_config;

  SparseDistanceCOOSPMVInputs<value_idx, value_t> params;
};

const std::vector<SparseDistanceCOOSPMVInputs<int, float>> inputs_i32_f = {
  {2,
   {0, 2, 4, 6, 8},
   {0, 1, 0, 1, 0, 1, 0, 1},
   {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f},
   {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    5.0},
   raft::distance::DistanceType::InnerProduct},
  {2,
   {0, 2, 4, 6, 8},
   {0, 1, 0, 1, 0, 1, 0, 1},  // indices
   {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f},
   {
     // dense output
     0.0,
     4.0,
     3026.0,
     226.0,
     4.0,
     0.0,
     2930.0,
     234.0,
     3026.0,
     2930.0,
     0.0,
     1832.0,
     226.0,
     234.0,
     1832.0,
     0.0,
   },
   raft::distance::DistanceType::L2Unexpanded},

  {10,
   {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
   {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2,
    3, 4, 7, 0, 1, 2, 3, 4, 6, 8, 0, 1, 2, 5, 7, 1, 5,
    8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
   {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131,
    0.5167, 0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190,
    0.8206, 0.3625, 0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431,
    0.0535, 0.2225, 0.8853, 0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,
    0.5279, 0.4885, 0.3495, 0.5079, 0.2325, 0.2331, 0.3018, 0.6231, 0.2645,
    0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
   {0.0,
    3.3954660629919076,
    5.6469232737388815,
    6.373112846266441,
    4.0212880272531715,
    6.916281504639404,
    5.741508386786526,
    5.411470999663036,
    9.0,
    4.977014354725805,
    3.3954660629919076,
    0.0,
    7.56256082439209,
    5.540261147481582,
    4.832322929216881,
    4.62003193872216,
    6.498056792320361,
    4.309846252268695,
    6.317531174829905,
    6.016362684141827,
    5.6469232737388815,
    7.56256082439209,
    0.0,
    5.974878731322299,
    4.898357301336036,
    6.442097410320605,
    5.227077347287883,
    7.134101195584642,
    5.457753923371659,
    7.0,
    6.373112846266441,
    5.540261147481582,
    5.974878731322299,
    0.0,
    5.5507273748583,
    4.897749658726415,
    9.0,
    8.398776718824767,
    3.908281400328807,
    4.83431066343688,
    4.0212880272531715,
    4.832322929216881,
    4.898357301336036,
    5.5507273748583,
    0.0,
    6.632989819428174,
    7.438852294822894,
    5.6631570310967465,
    7.579428202635459,
    6.760811985364303,
    6.916281504639404,
    4.62003193872216,
    6.442097410320605,
    4.897749658726415,
    6.632989819428174,
    0.0,
    5.249404187382862,
    6.072559523278559,
    4.07661278488929,
    6.19678948003145,
    5.741508386786526,
    6.498056792320361,
    5.227077347287883,
    9.0,
    7.438852294822894,
    5.249404187382862,
    0.0,
    3.854811639654704,
    6.652724827169063,
    5.298236851430971,
    5.411470999663036,
    4.309846252268695,
    7.134101195584642,
    8.398776718824767,
    5.6631570310967465,
    6.072559523278559,
    3.854811639654704,
    0.0,
    7.529184598969917,
    6.903282911791188,
    9.0,
    6.317531174829905,
    5.457753923371659,
    3.908281400328807,
    7.579428202635459,
    4.07661278488929,
    6.652724827169063,
    7.529184598969917,
    0.0,
    7.0,
    4.977014354725805,
    6.016362684141827,
    7.0,
    4.83431066343688,
    6.760811985364303,
    6.19678948003145,
    5.298236851430971,
    6.903282911791188,
    7.0,
    0.0},
   raft::distance::DistanceType::Canberra},

  {10,
   {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
   {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2,
    3, 4, 7, 0, 1, 2, 3, 4, 6, 8, 0, 1, 2, 5, 7, 1, 5,
    8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
   {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131,
    0.5167, 0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190,
    0.8206, 0.3625, 0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431,
    0.0535, 0.2225, 0.8853, 0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,
    0.5279, 0.4885, 0.3495, 0.5079, 0.2325, 0.2331, 0.3018, 0.6231, 0.2645,
    0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
   {0.0,
    1.31462855332296,
    1.3690307816129905,
    1.698603990921237,
    1.3460470789553531,
    1.6636670712582544,
    1.2651744044972217,
    1.1938329352055201,
    1.8811409082590185,
    1.3653115050624267,
    1.31462855332296,
    0.0,
    1.9447722703291133,
    1.42818777206562,
    1.4685491458946494,
    1.3071999866010466,
    1.4988622861692171,
    0.9698559287406783,
    1.4972023224597841,
    1.5243383567266802,
    1.3690307816129905,
    1.9447722703291133,
    0.0,
    1.2748400840107568,
    1.0599569946448246,
    1.546591282841402,
    1.147526531928459,
    1.447002179128145,
    1.5982242387673176,
    1.3112533607072414,
    1.698603990921237,
    1.42818777206562,
    1.2748400840107568,
    0.0,
    1.038121552545461,
    1.011788365364402,
    1.3907391109256988,
    1.3128200942311496,
    1.19595706584447,
    1.3233328139624725,
    1.3460470789553531,
    1.4685491458946494,
    1.0599569946448246,
    1.038121552545461,
    0.0,
    1.3642741698145529,
    1.3493868683808095,
    1.394942694628328,
    1.572881849642552,
    1.380122665319464,
    1.6636670712582544,
    1.3071999866010466,
    1.546591282841402,
    1.011788365364402,
    1.3642741698145529,
    0.0,
    1.018961640373018,
    1.0114394258945634,
    0.8338711034820684,
    1.1247823842299223,
    1.2651744044972217,
    1.4988622861692171,
    1.147526531928459,
    1.3907391109256988,
    1.3493868683808095,
    1.018961640373018,
    0.0,
    0.7701238110357329,
    1.245486437864406,
    0.5551259549534626,
    1.1938329352055201,
    0.9698559287406783,
    1.447002179128145,
    1.3128200942311496,
    1.394942694628328,
    1.0114394258945634,
    0.7701238110357329,
    0.0,
    1.1886800117391216,
    1.0083692448135637,
    1.8811409082590185,
    1.4972023224597841,
    1.5982242387673176,
    1.19595706584447,
    1.572881849642552,
    0.8338711034820684,
    1.245486437864406,
    1.1886800117391216,
    0.0,
    1.3661374102525012,
    1.3653115050624267,
    1.5243383567266802,
    1.3112533607072414,
    1.3233328139624725,
    1.380122665319464,
    1.1247823842299223,
    0.5551259549534626,
    1.0083692448135637,
    1.3661374102525012,
    0.0},
   raft::distance::DistanceType::LpUnexpanded,
   2.0},

  {10,
   {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
   {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2,
    3, 4, 7, 0, 1, 2, 3, 4, 6, 8, 0, 1, 2, 5, 7, 1, 5,
    8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
   {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131,
    0.5167, 0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190,
    0.8206, 0.3625, 0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431,
    0.0535, 0.2225, 0.8853, 0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,
    0.5279, 0.4885, 0.3495, 0.5079, 0.2325, 0.2331, 0.3018, 0.6231, 0.2645,
    0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
   {0.0,
    0.9251771844789913,
    0.9036452083899731,
    0.9251771844789913,
    0.8706483735804971,
    0.9251771844789913,
    0.717493881903289,
    0.6920214832303888,
    0.9251771844789913,
    0.9251771844789913,
    0.9251771844789913,
    0.0,
    0.9036452083899731,
    0.8655339692155823,
    0.8706483735804971,
    0.8655339692155823,
    0.8655339692155823,
    0.6329837991017668,
    0.8655339692155823,
    0.8655339692155823,
    0.9036452083899731,
    0.9036452083899731,
    0.0,
    0.7988276152181608,
    0.7028075145996631,
    0.9036452083899731,
    0.9036452083899731,
    0.9036452083899731,
    0.8429599432532096,
    0.9036452083899731,
    0.9251771844789913,
    0.8655339692155823,
    0.7988276152181608,
    0.0,
    0.48376552205293305,
    0.8206394616536681,
    0.8206394616536681,
    0.8206394616536681,
    0.8429599432532096,
    0.8206394616536681,
    0.8706483735804971,
    0.8706483735804971,
    0.7028075145996631,
    0.48376552205293305,
    0.0,
    0.8706483735804971,
    0.8706483735804971,
    0.8706483735804971,
    0.8429599432532096,
    0.8706483735804971,
    0.9251771844789913,
    0.8655339692155823,
    0.9036452083899731,
    0.8206394616536681,
    0.8706483735804971,
    0.0,
    0.8853924473642432,
    0.535821510936138,
    0.6497196601457607,
    0.8853924473642432,
    0.717493881903289,
    0.8655339692155823,
    0.9036452083899731,
    0.8206394616536681,
    0.8706483735804971,
    0.8853924473642432,
    0.0,
    0.5279604218147174,
    0.6658348373853169,
    0.33799874888632914,
    0.6920214832303888,
    0.6329837991017668,
    0.9036452083899731,
    0.8206394616536681,
    0.8706483735804971,
    0.535821510936138,
    0.5279604218147174,
    0.0,
    0.662579808115858,
    0.5079750812968089,
    0.9251771844789913,
    0.8655339692155823,
    0.8429599432532096,
    0.8429599432532096,
    0.8429599432532096,
    0.6497196601457607,
    0.6658348373853169,
    0.662579808115858,
    0.0,
    0.8429599432532096,
    0.9251771844789913,
    0.8655339692155823,
    0.9036452083899731,
    0.8206394616536681,
    0.8706483735804971,
    0.8853924473642432,
    0.33799874888632914,
    0.5079750812968089,
    0.8429599432532096,
    0.0},
   raft::distance::DistanceType::Linf},

  {4,
   {0, 1, 1, 2, 4},
   {3, 2, 0, 1},  // indices
   {0.99296, 0.42180, 0.11687, 0.305869},
   {
     // dense output
     0.0,
     0.99296,
     1.41476,
     1.415707,
     0.99296,
     0.0,
     0.42180,
     0.42274,
     1.41476,
     0.42180,
     0.0,
     0.84454,
     1.41570,
     0.42274,
     0.84454,
     0.0,
   },
   raft::distance::DistanceType::L1}

};

typedef SparseDistanceCOOSPMVTest<int, float> SparseDistanceCOOSPMVTestF;
TEST_P(SparseDistanceCOOSPMVTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseDistanceCOOSPMVTests, SparseDistanceCOOSPMVTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // namespace distance
};  // end namespace sparse
};  // end namespace raft
