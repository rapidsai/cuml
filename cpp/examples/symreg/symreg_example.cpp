/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <cuml/genetic/common.h>
#include <cuml/genetic/genetic.h>
#include <cuml/genetic/program.h>
#include <cuml/cuml.hpp>
#include <raft/mr/device/allocator.hpp>
#include <cuml/common/logger.hpp>

// Namspace alias
namespace cg = cuml::genetic;

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg,
             const T default_val) {
  T argval = default_val;
  char **itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char **begin, char **end, const std::string &arg) {
  char **itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

int main(int argc, char* argv[]){

  // View debug messages
  ML::Logger::get().setLevel(CUML_LEVEL_DEBUG);

  // Training hyper parameters(contains default vals)
  cg::param params;

  // Process training arguments
  const int population_size   = get_argval(argv,argv+argc,"-population_size",params.population_size);
  const uint64_t random_state = get_argval(argv,argv+argc,"-random_state",params.random_state);
  const int num_generations   = get_argval(argv,argv+argc,"-generations",params.generations);
  const float stop_criterion  = get_argval(argv,argv+argc,"-stopping_criteria",params.stopping_criteria);
  const float p_crossover     = get_argval(argv,argv+argc,"-p_crossover",params.p_crossover);
  const float p_subtree       = get_argval(argv,argv+argc,"-p_subtree",params.p_subtree_mutation);
  const float p_hoist         = get_argval(argv,argv+argc,"-p_hoist",params.p_hoist_mutation);
  const float p_point         = get_argval(argv,argv+argc,"-p_point",params.p_point_mutation);
  const float p_point_replace = get_argval(argv,argv+argc,"-p_point_replace",params.p_point_replace);
  const float parsimony_coeff = get_argval(argv,argv+argc,"-parsimony_coeff",params.parsimony_coefficient);
  const std::string metric    = get_argval(argv,argv+argc,"-metric",std::string("mae")); // mean absolute error is default

  // Process dataset specific arguments
  const int n_cols            = get_argval(argv,argv+argc,"-n_cols",0);
  const int n_train_rows      = get_argval(argv,argv+argc,"-n_train_rows",0);
  const int n_test_rows       = get_argval(argv,argv+argc,"-n_test_rows",0);

  const std::string train_data    = get_argval(argv,argv+argc,"-train_data",std::string("train_data.txt"));
  const std::string train_labels  = get_argval(argv,argv+argc,"-train_labels",std::string("train_labels.txt"));
  const std::string test_data     = get_argval(argv,argv+argc,"-test_data",std::string("test_data.txt"));
  const std::string test_labels   = get_argval(argv,argv+argc,"-test_labels",std::string("test_labels.txt"));

  // Optionally accept files containing sample weights - if none are specified, then we consider a uniform distribution
  const std::string train_weights = get_argval(argv,argv+argc,"-train_weights",std::string(""));
  const std::string test_weights  = get_argval(argv,argv+argc,"-test_weights",std::string(""));

  std::vector<float> h_traindata;
  std::vector<float> h_testdata;
  std::vector<float> h_trainlabels;
  std::vector<float> h_testlabels;
  std::vector<float> h_trainweights;
  std::vector<float> h_testweights;

  // Read dataset inputs
  {
    float val = 0.0f;
    // Read train dataset
    std::ifstream input_stream(train_data, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << train_data << std::endl;
      return 1;
    }
    
    std::cout << "Reading input with " << n_train_rows << " rows and " << n_cols
              << " columns from " << train_data << "." << std::endl;
    
    h_traindata.reserve(n_train_rows * n_cols);
    while (input_stream >> val) {
      h_traindata.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
    
    // Read training labels
    input_stream.open(train_labels, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << train_labels << std::endl;
      return 1;
    }
    
    std::cout << "Reading input with " << n_train_rows << " rows "
              << "from " << train_labels << "." << std::endl;
    
    h_trainlabels.reserve(n_train_rows);
    while (input_stream >> val) {
      h_trainlabels.push_back(val);
    }
    input_stream.close();
    input_stream.clear();

    // Read test dataset
    input_stream.open(test_data, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << test_data << std::endl;
      return 1;
    }
    
    std::cout << "Reading input with " << n_test_rows << " rows and " << n_cols
              << " columns from " << test_data << "." << std::endl;
    
    h_testdata.reserve(n_test_rows * n_cols);
    while (input_stream >> val) {
      h_testdata.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
    
    // Read test labels
    input_stream.open(test_labels, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << test_labels << std::endl;
      return 1;
    }
    
    std::cout << "Reading input with " << n_test_rows << " rows"
              << " from " << test_labels << "." << std::endl;
    
    h_traindata.reserve(n_test_rows);
    while (input_stream >> val) {
      h_testlabels.push_back(val);
    }
    input_stream.close();
    input_stream.clear();

    // Read training weights
    if(train_weights != ""){
      input_stream.open(train_weights,std::ios::in);
      if(!input_stream.is_open()){
        std::cerr << "ERROR: Could not open input file " << train_weights << std::endl;
        return 1;
      }
      
      std::cout << "Reading input with " << n_train_rows << " rows"
                << " from " << train_weights << "." << std::endl;
      h_trainweights.reserve(n_train_rows);
      while (input_stream >> val) {
        h_trainweights.push_back(val);
      }
      input_stream.close();
      input_stream.clear();
    }
    else{
      h_trainweights.resize(n_train_rows,1.0f);
    }

    // Read test weights
    if(test_weights != ""){
      input_stream.open(test_weights,std::ios::in);
      if(!input_stream.is_open()){
        std::cerr << "ERROR: Could not open input file " << test_weights << std::endl;
        return 1;
      }
      
      std::cout << "Reading input with " << n_test_rows << " rows"
                << " from " << test_weights << "." << std::endl;
      h_testweights.reserve(n_test_rows);
      while (input_stream >> val) {
        h_testweights.push_back(val);
      }
      input_stream.close();
      input_stream.clear();
    }
    else{
      h_testweights.resize(n_test_rows,1.0f);
    }
  }
  
  // Check for valid mutation probability distribution
  float p_sum = p_crossover + p_hoist + p_point + p_subtree;
  if(p_sum >= 1.0f || p_sum <= 0.0f){
    std::cerr << "ERROR: Invalid mutation probabilities provided" << std::endl 
              << "Probability sum for crossover, subtree, host and point mutations is " << p_sum << std::endl;
    return 1;
  }

  // Check if point_replace < 1.0f
  if(p_point_replace > 1.0f || p_point_replace < 0.0f){
    std::cerr << "ERROR: Invalid value for point replacement probability" << std::endl ;
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
  if(metric == "mae"){
    params.metric = cg::metric_t::mae;
  }else if(metric == "mse"){
    params.metric = cg::metric_t::mse;
  }else if(metric == "rmse"){
    params.metric = cg::metric_t::rmse;
  }else{
    std::cerr << "ERROR: Invalid metric specified for regression (can only be mae, mse or rmse) " << std::endl;
    return 1;
  }

  // Initialize and allocate device memory
  raft::handle_t handle;
  std::shared_ptr<ML::deviceAllocator> allocator(
      new raft::mr::device::default_allocator());

  handle.set_device_allocator(allocator);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);

  float* d_traindata    = nullptr;
  float* d_trainlabels  = nullptr;
  float* d_trainweights = nullptr;
  float* d_testdata     = nullptr;
  float* d_testlabels   = nullptr;
  float* d_testweights  = nullptr;
  float* d_predlabels   = nullptr;
  float* d_score        = nullptr;

  cg::program_t d_finalprogs;  // pointer to last generation ASTs on device

  CUDA_RT_CALL(cudaMalloc(&d_traindata,sizeof(float)*n_cols*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_traindata,h_traindata.data(),
                                sizeof(float)*n_cols*n_train_rows,
                                cudaMemcpyHostToDevice,stream));
  
  CUDA_RT_CALL(cudaMalloc(&d_trainlabels,sizeof(float)*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_trainlabels,h_trainlabels.data(),
                               sizeof(float)*n_train_rows,
                               cudaMemcpyHostToDevice,stream)); 

  CUDA_RT_CALL(cudaMalloc(&d_trainweights,sizeof(float)*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_trainweights,h_trainweights.data(),
                               sizeof(float)*n_train_rows,
                               cudaMemcpyHostToDevice,stream)); 

  CUDA_RT_CALL(cudaMalloc(&d_testdata,sizeof(float)*n_cols*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testdata,h_testdata.data(),
                                sizeof(float)*n_cols*n_test_rows,
                                cudaMemcpyHostToDevice,stream));
  
  CUDA_RT_CALL(cudaMalloc(&d_testlabels,sizeof(float)*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testlabels,h_testlabels.data(),
                               sizeof(float)*n_test_rows,
                               cudaMemcpyHostToDevice,stream));

  CUDA_RT_CALL(cudaMalloc(&d_testweights,sizeof(float)*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testweights,h_testweights.data(),
                               sizeof(float)*n_test_rows,
                               cudaMemcpyHostToDevice,stream)); 
  
  CUDA_RT_CALL(cudaMalloc(&d_predlabels,sizeof(float)*n_test_rows));  
  CUDA_RT_CALL(cudaMalloc(&d_score,sizeof(float)));
  
  // Initialize AST
  d_finalprogs = (cg::program_t)handle.get_device_allocator()->allocate(params.population_size*sizeof(cg::program),stream);

  std::vector<std::vector<cg::program>> history;
  history.reserve(params.generations);
  
  // Begin training
  std::cout << "Beginning training on given dataset" << std::endl;
  cg::symFit(handle,d_traindata,d_trainlabels,d_trainweights,n_train_rows,n_cols,params,d_finalprogs,history);

  int n_gen = history.size();
  std::cout << "Finished training for " << n_gen << " generations." << std::endl;
  
  // Find index of best program
  int best_idx = 0;
  float opt_fitness = history[n_gen-1][0].raw_fitness_;

  // For all 3 loss functions - min is better
  for (int i=1; i<params.population_size ; ++i){
    if(history[n_gen-1][i].raw_fitness_ < opt_fitness){
      best_idx = i;
      opt_fitness = history[n_gen-1][i].raw_fitness_;
    }
  }

  std::string eqn = cg::stringify(history[n_gen-1][best_idx]);
  std::cout << "Best AST index is : " << best_idx << std::endl;
  std::cout << "Best AST depth : " << cg::get_depth(history[n_gen-1][best_idx]) << std::endl;
  std::cout << "Best AST length : " << history[n_gen-1][best_idx].len << std::endl;
  std::cout << "Best AST equation is : " << eqn << std::endl;
  
  // Predict values for test dataset
  cuml::genetic::symRegPredict(handle,d_testdata,n_test_rows,d_finalprogs+best_idx,d_predlabels);

  std::vector<float> h_predlabels(n_test_rows,0.0f);

  CUDA_RT_CALL(cudaMemcpy(h_predlabels.data(),d_predlabels,n_test_rows * sizeof(float),cudaMemcpyDeviceToHost));
  
  std::cout << "Some Predicted test values:" << std::endl;
  std::copy(h_predlabels.begin(),h_predlabels.end(),std::ostream_iterator<float>(std::cout,";"));
  std::cout << std::endl;

  std::cout << "Some Actual test values vals:" << std::endl;
  std::copy(h_testlabels.begin(),h_testlabels.end(),std::ostream_iterator<float>(std::cout,";"));
  std::cout << std::endl;

  // Output fitness score
  cuml::genetic::compute_metric(handle,n_test_rows,1,d_testlabels,d_predlabels,d_testweights,d_score,params);

  float h_score = 0.0f;
  CUDA_RT_CALL(cudaMemcpyAsync(&h_score,d_score,sizeof(float),cudaMemcpyDeviceToHost,stream));
  
  std::cout << " Metric Score for test set : " << h_score << std::endl;
  
  // Free up device memory
  CUDA_RT_CALL(cudaFree(d_traindata));
  CUDA_RT_CALL(cudaFree(d_trainlabels));
  CUDA_RT_CALL(cudaFree(d_trainweights));
  CUDA_RT_CALL(cudaFree(d_testdata));
  CUDA_RT_CALL(cudaFree(d_testlabels));
  CUDA_RT_CALL(cudaFree(d_testweights));
  CUDA_RT_CALL(cudaFree(d_predlabels));
  CUDA_RT_CALL(cudaFree(d_score));
  handle.get_device_allocator()->deallocate(d_finalprogs,
                                sizeof(cuml::genetic::program) * params.population_size, stream);
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  return 0;
}
