/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/genetic/common.h>
#include <cuml/genetic/genetic.h>
#include <cuml/genetic/program.h>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

// Namespace alias
namespace cg = cuml::genetic;

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call,                                                          \
              __LINE__,                                                       \
              __FILE__,                                                       \
              cudaGetErrorString(cudaStatus),                                 \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val)
{
  T argval   = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

template <typename math_t = float>
int parse_col_major(const std::string fname,
                    std::vector<math_t>& vec,
                    const int n_rows,
                    const int n_cols)
{
  std::ifstream is(fname);
  if (!is.is_open()) {
    std::cerr << "ERROR: Could not open file " << fname << std::endl;
    return 1;
  }

  std::istream_iterator<math_t> start(is), end;
  vec.reserve(n_rows * n_cols);
  vec.assign(start, end);
  return 0;
}

int main(int argc, char* argv[])
{
  // Training hyper parameters(contains default vals)
  cg::param params;

  // Cuda Events to track execution time for various components
  cudaEvent_t start, stop;
  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));

  // Process training arguments
  const int population_size =
    get_argval(argv, argv + argc, "-population_size", params.population_size);
  const uint64_t random_state = get_argval(argv, argv + argc, "-random_state", params.random_state);
  const int num_generations   = get_argval(argv, argv + argc, "-generations", params.generations);
  const float stop_criterion =
    get_argval(argv, argv + argc, "-stopping_criteria", params.stopping_criteria);
  const float p_crossover = get_argval(argv, argv + argc, "-p_crossover", params.p_crossover);
  const float p_subtree   = get_argval(argv, argv + argc, "-p_subtree", params.p_subtree_mutation);
  const float p_hoist     = get_argval(argv, argv + argc, "-p_hoist", params.p_hoist_mutation);
  const float p_point     = get_argval(argv, argv + argc, "-p_point", params.p_point_mutation);
  const float p_point_replace =
    get_argval(argv, argv + argc, "-p_point_replace", params.p_point_replace);
  const float parsimony_coeff =
    get_argval(argv, argv + argc, "-parsimony_coeff", params.parsimony_coefficient);
  const std::string metric = get_argval(argv,
                                        argv + argc,
                                        "-metric",
                                        std::string("mae"));  // mean absolute error is default

  // Process dataset specific arguments
  const int n_cols       = get_argval(argv, argv + argc, "-n_cols", 0);
  const int n_train_rows = get_argval(argv, argv + argc, "-n_train_rows", 0);
  const int n_test_rows  = get_argval(argv, argv + argc, "-n_test_rows", 0);

  const std::string fX_train =
    get_argval(argv, argv + argc, "-train_data", std::string("train_data.txt"));
  const std::string fy_train =
    get_argval(argv, argv + argc, "-train_labels", std::string("train_labels.txt"));
  const std::string fX_test =
    get_argval(argv, argv + argc, "-test_data", std::string("test_data.txt"));
  const std::string fy_test =
    get_argval(argv, argv + argc, "-test_labels", std::string("test_labels.txt"));

  // Optionally accept files containing sample weights - if none are specified, then we consider a
  // uniform distribution
  const std::string fw_train =
    get_argval(argv, argv + argc, "-train_weights", std::string("train_weights.txt"));
  const std::string fw_test =
    get_argval(argv, argv + argc, "-test_weights", std::string("test_weights.txt"));

  std::vector<float> X_train;
  std::vector<float> X_test;
  std::vector<float> y_train;
  std::vector<float> y_test;
  std::vector<float> w_train;
  std::vector<float> w_test;

  // Read input
  if (parse_col_major(fX_train, X_train, n_train_rows, n_cols)) return 1;
  if (parse_col_major(fX_test, X_test, n_test_rows, n_cols)) return 1;
  if (parse_col_major(fy_train, y_train, n_train_rows, 1)) return 1;
  if (parse_col_major(fy_test, y_test, n_test_rows, 1)) return 1;
  if (parse_col_major(fw_train, w_train, n_train_rows, 1)) {
    std::cerr << "Defaulting to uniform training weights" << std::endl;
    w_train.resize(n_train_rows, 1.0f);
  }
  if (parse_col_major(fw_test, w_test, n_test_rows, 1)) {
    std::cerr << "Defaulting to uniform test weights" << std::endl;
    w_test.resize(n_test_rows, 1.0f);
  }

  // Check for valid mutation probability distribution
  float p_sum = p_crossover + p_hoist + p_point + p_subtree;
  if (p_sum >= 1.0f || p_sum <= 0.0f) {
    std::cerr << "ERROR: Invalid mutation probabilities provided" << std::endl
              << "Probability sum for crossover, subtree, host and point mutations is " << p_sum
              << std::endl;
    return 1;
  }

  // Check if point_replace < 1.0f
  if (p_point_replace > 1.0f || p_point_replace < 0.0f) {
    std::cerr << "ERROR: Invalid value for point replacement probability" << std::endl;
    return 1;
  }

  // Set all training parameters
  params.num_features          = n_cols;
  params.population_size       = population_size;
  params.random_state          = random_state;
  params.generations           = num_generations;
  params.stopping_criteria     = stop_criterion;
  params.p_crossover           = p_crossover;
  params.p_subtree_mutation    = p_subtree;
  params.p_hoist_mutation      = p_hoist;
  params.p_point_mutation      = p_point;
  params.p_point_replace       = p_point_replace;
  params.parsimony_coefficient = parsimony_coeff;

  // Set training metric
  if (metric == "mae") {
    params.metric = cg::metric_t::mae;
  } else if (metric == "mse") {
    params.metric = cg::metric_t::mse;
  } else if (metric == "rmse") {
    params.metric = cg::metric_t::rmse;
  } else {
    std::cerr << "ERROR: Invalid metric specified for regression (can only be "
                 "mae, mse or rmse) "
              << std::endl;
    return 1;
  }

  /* ======================= Begin GPU memory allocation ======================= */
  std::cout << "***************************************" << std::endl;

  cudaStream_t stream;
  raft::handle_t handle{stream};

  // Begin recording time
  cudaEventRecord(start, stream);

  rmm::device_uvector<float> dX_train(n_cols * n_train_rows, stream);
  rmm::device_uvector<float> dy_train(n_train_rows, stream);
  rmm::device_uvector<float> dw_train(n_train_rows, stream);
  rmm::device_uvector<float> dX_test(n_cols * n_test_rows, stream);
  rmm::device_uvector<float> dy_test(n_test_rows, stream);
  rmm::device_uvector<float> dw_test(n_test_rows, stream);
  rmm::device_uvector<float> dy_pred(n_test_rows, stream);
  rmm::device_scalar<float> d_score{stream};

  cg::program_t d_finalprogs;  // pointer to last generation ASTs on device

  CUDA_RT_CALL(cudaMemcpyAsync(dX_train.data(),
                               X_train.data(),
                               sizeof(float) * dX_train.size(),
                               cudaMemcpyHostToDevice,
                               stream));

  CUDA_RT_CALL(cudaMemcpyAsync(dy_train.data(),
                               y_train.data(),
                               sizeof(float) * dy_train.size(),
                               cudaMemcpyHostToDevice,
                               stream));

  CUDA_RT_CALL(cudaMemcpyAsync(dw_train.data(),
                               w_train.data(),
                               sizeof(float) * dw_train.size(),
                               cudaMemcpyHostToDevice,
                               stream));

  CUDA_RT_CALL(cudaMemcpyAsync(
    dX_test.data(), X_test.data(), sizeof(float) * dX_test.size(), cudaMemcpyHostToDevice, stream));

  CUDA_RT_CALL(cudaMemcpyAsync(
    dy_test.data(), y_test.data(), sizeof(float) * dy_test.size(), cudaMemcpyHostToDevice, stream));

  CUDA_RT_CALL(cudaMemcpyAsync(
    dw_test.data(), w_test.data(), sizeof(float) * n_test_rows, cudaMemcpyHostToDevice, stream));

  // Initialize AST
  auto curr_mr = rmm::mr::get_current_device_resource();
  d_finalprogs = static_cast<cg::program_t>(curr_mr->allocate(params.population_size, stream));

  std::vector<std::vector<cg::program>> history;
  history.reserve(params.generations);

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float alloc_time;
  cudaEventElapsedTime(&alloc_time, start, stop);

  std::cout << "Allocated device memory in " << std::setw(10) << alloc_time << "ms" << std::endl;

  /* ======================= Begin training ======================= */

  std::cout << "***************************************" << std::endl;
  std::cout << std::setw(30) << "Beginning training for " << std::setw(15) << params.generations
            << " generations" << std::endl;
  cudaEventRecord(start, stream);

  cg::symFit(handle,
             dX_train.data(),
             dy_train.data(),
             dw_train.data(),
             n_train_rows,
             n_cols,
             params,
             d_finalprogs,
             history);

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float training_time;
  cudaEventElapsedTime(&training_time, start, stop);

  int n_gen = params.num_epochs;
  std::cout << std::setw(30) << "Convergence achieved in " << std::setw(15) << n_gen
            << " generations." << std::endl;

  // Find index of best program
  int best_idx      = 0;
  float opt_fitness = history.back()[0].raw_fitness_;

  // For all 3 loss functions - min is better
  for (int i = 1; i < params.population_size; ++i) {
    if (history.back()[i].raw_fitness_ < opt_fitness) {
      best_idx    = i;
      opt_fitness = history.back()[i].raw_fitness_;
    }
  }

  std::string eqn = cg::stringify(history.back()[best_idx]);
  std::cout << std::setw(30) << "Best AST depth " << std::setw(15) << history.back()[best_idx].depth
            << std::endl;
  std::cout << std::setw(30) << "Best AST length " << std::setw(15) << history.back()[best_idx].len
            << std::endl;
  std::cout << std::setw(30) << "Best AST equation " << std::setw(15) << eqn << std::endl;
  std::cout << "Training time = " << training_time << "ms" << std::endl;

  /* ======================= Begin testing ======================= */

  std::cout << "***************************************" << std::endl;
  std::cout << "Beginning Inference on test dataset " << std::endl;
  cudaEventRecord(start, stream);
  cuml::genetic::symRegPredict(
    handle, dX_test.data(), n_test_rows, d_finalprogs + best_idx, dy_pred.data());

  std::vector<float> hy_pred(n_test_rows, 0.0f);
  CUDA_RT_CALL(cudaMemcpy(
    hy_pred.data(), dy_pred.data(), n_test_rows * sizeof(float), cudaMemcpyDeviceToHost));

  cuml::genetic::compute_metric(
    handle, n_test_rows, 1, dy_test.data(), dy_pred.data(), dw_test.data(), d_score.data(), params);

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float inference_time;
  cudaEventElapsedTime(&inference_time, start, stop);

  // Output fitness score
  std::cout << "Inference score = " << d_score.value(stream) << std::endl;
  std::cout << "Inference time = " << inference_time << "ms" << std::endl;

  std::cout << "Some Predicted test values:" << std::endl;
  std::copy(hy_pred.begin(), hy_pred.begin() + 5, std::ostream_iterator<float>(std::cout, ";"));
  std::cout << std::endl;

  std::cout << "Corresponding Actual test values:" << std::endl;
  std::copy(y_test.begin(), y_test.begin() + 5, std::ostream_iterator<float>(std::cout, ";"));
  std::cout << std::endl;

  /* ======================= Reset data ======================= */

  curr_mr->deallocate(d_finalprogs, params.population_size, stream);
  CUDA_RT_CALL(cudaEventDestroy(start));
  CUDA_RT_CALL(cudaEventDestroy(stop));
  return 0;
}
