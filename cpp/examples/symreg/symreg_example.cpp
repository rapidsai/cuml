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
  ML::Logger::get().setLevel(CUML_LEVEL_DEBUG);

  // Get all command line arguments and input
  const int n_cols = get_argval(argv,argv+argc,"-n_cols",0);
  const int n_train_rows = get_argval(argv,argv+argc,"-n_train_rows",0);
  const int n_test_rows = get_argval(argv,argv+argc,"-n_test_rows",0);
  const int population_size = get_argval(argv,argv+argc,"-population_size",100);
  const int random_state = get_argval(argv,argv+argc,"-random_state",2021);

  const std::string train_data = get_argval(argv,argv+argc,"-train_data",std::string(""));
  const std::string train_labels = get_argval(argv,argv+argc,"-train_labels",std::string(""));
  const std::string test_data = get_argval(argv,argv+argc,"-test_data",std::string(""));
  const std::string test_labels = get_argval(argv,argv+argc,"-test_labels",std::string(""));

  std::vector<float> h_traindata;
  std::vector<float> h_trainlabels;
  std::vector<float> h_trainweights(n_train_rows,1.0f);
  std::vector<float> h_testdata;
  std::vector<float> h_testlabels;
  std::vector<float> h_testweights(n_test_rows,1.0f);

  // Read train dataset
  if("" != train_data){
    std::ifstream input_stream(train_data, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << train_data << std::endl;
      return 1;
    }
    std::cout << "Reading input with " << n_train_rows << " rows and " << n_cols
              << " columns from " << train_data << "." << std::endl;
    h_traindata.reserve(n_train_rows * n_cols);
    float val = 0.0;
    while (input_stream >> val) {
      h_traindata.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
  }
  else{
    std::cerr << "ERROR: Training dataset not specified" << std::endl;
    return 1;
  }

  // Read training labels
  if("" != train_labels){
    std::ifstream input_stream(train_labels, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << train_labels << std::endl;
      return 1;
    }
    std::cout << "Reading input with " << n_train_rows << " rows "
              << "from " << train_labels << "." << std::endl;
    h_trainlabels.reserve(n_train_rows);
    float val = 0.0;
    while (input_stream >> val) {
      h_trainlabels.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
  }
  else{
    std::cerr << "ERROR: Training labels not specified" << std::endl;
    return 1;
  }

  // Read test dataset
  if("" != test_data){
    std::ifstream input_stream(test_data, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << test_data << std::endl;
      return 1;
    }
    std::cout << "Reading input with " << n_test_rows << " rows and " << n_cols
              << " columns from " << test_data << "." << std::endl;
    h_testdata.reserve(n_test_rows * n_cols);
    float val = 0.0;
    while (input_stream >> val) {
      h_testdata.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
  }
  else{
    std::cerr << "ERROR: Test dataset not specified" << std::endl;
    return 1;
  }
  // Read test labels
  if("" != test_labels){
    std::ifstream input_stream(test_labels, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << test_labels << std::endl;
      return 1;
    }
    std::cout << "Reading input with " << n_test_rows << " rows"
              << " from " << test_labels << "." << std::endl;
    h_traindata.reserve(n_test_rows);
    float val = 0.0;
    while (input_stream >> val) {
      h_testlabels.push_back(val);
    }
    input_stream.close();
    input_stream.clear();
  }
  else{
    std::cerr << "ERROR: Test labels not specified" << std::endl;
    return 1;
  }

  // Set training parameters
  cuml::genetic::param params;
  params.population_size = population_size;
  params.num_features = n_cols;
  params.random_state = random_state;

  // Allocate device memory
  raft::handle_t handle;

  std::shared_ptr<ML::deviceAllocator> allocator(
      new raft::mr::device::default_allocator());

  handle.set_device_allocator(allocator);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);

  float* d_traindata = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_traindata,sizeof(float)*n_cols*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_traindata,h_traindata.data(),
                                sizeof(float)*n_cols*n_train_rows,
                                cudaMemcpyHostToDevice,stream));
  
  float* d_trainlabels = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_trainlabels,sizeof(float)*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_trainlabels,h_trainlabels.data(),
                               sizeof(float)*n_train_rows,
                               cudaMemcpyHostToDevice,stream)); 
  float* d_trainweights = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_trainweights,sizeof(float)*n_train_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_trainweights,h_trainweights.data(),
                               sizeof(float)*n_train_rows,
                               cudaMemcpyHostToDevice,stream)); 

  float* d_testdata = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_testdata,sizeof(float)*n_cols*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testdata,h_testdata.data(),
                                sizeof(float)*n_cols*n_test_rows,
                                cudaMemcpyHostToDevice,stream));
  
  float* d_testlabels = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_testlabels,sizeof(float)*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testlabels,h_testlabels.data(),
                               sizeof(float)*n_test_rows,
                               cudaMemcpyHostToDevice,stream));

  float* d_testweights = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_testweights,sizeof(float)*n_test_rows));
  CUDA_RT_CALL(cudaMemcpyAsync(d_testweights,h_testweights.data(),
                               sizeof(float)*n_test_rows,
                               cudaMemcpyHostToDevice,stream)); 
  
  float* d_predlabels = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_predlabels,sizeof(float)*n_test_rows));

  float* d_score = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_testlabels,sizeof(float)));
  
  // Initialize AST
  cuml::genetic::program_t final_progs;
  final_progs = (cuml::genetic::program_t)handle.get_device_allocator()->allocate(params.population_size*sizeof(cuml::genetic::program),stream);
  std::vector<std::vector<cuml::genetic::program>> history;
  history.reserve(params.generations);
  
  CUML_LOG_DEBUG("This is a debugging warning from the cuML logger!");

  // Begin training
  cuml::genetic::symFit(handle,d_traindata,d_trainlabels,d_trainweights,n_train_rows,n_cols,params,final_progs,history);
  
  for(int i = 0; i < params.population_size; ++i){
    std::cout <<" Program " << (i+1) <<" -> " 
              << cuml::genetic::stringify(history[params.generations-1][i]) << std::endl;
  }

  /*
  // Find index of best program
  int n_gen = history.size();
  auto best_prog = std::min_element(history[n_gen-1].begin(),history[n_gen-1].end(),
                      [params](const cuml::genetic::program &a, const cuml::genetic::program &b){
                        return cuml::genetic::fitness(a,params) < cuml::genetic::fitness(b,params);
                      });
  int best_idx = std::distance(history[n_gen-1].begin(),best_prog);

  // Predict values for test dataste
  cuml::genetic::symRegPredict(handle,d_testdata,n_test_rows,final_progs+best_idx,d_predlabels);

  // Output fitness score
  cuml::genetic::compute_metric(handle,n_test_rows,1,d_testlabels,d_predlabels,d_testweights,d_score,params);

  float h_score[1] = {0.0f};
  // CUDA_RT_CALL(cudaMemcpyAsync(h_score,d_score,sizeof(float),cudaMemcpyDeviceToHost,stream));

  std::cout << " Fitness score : " << best_idx << " "<< history[n_gen-1].size() << std::endl;
  std::string eqn = cuml::genetic::stringify(history[n_gen-1][best_idx]);
  std::cout << " Best program : " << eqn;
  */
  return 0;
}
