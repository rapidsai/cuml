/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cuml/genetic/node.h>
#include <cuml/genetic/common.h>
#include <cuml/genetic/program.h>
#include <raft/handle.hpp>
#include <test_utils.h>
#include <vector>
#include <iostream>

namespace cuml{
namespace genetic{

class GeneticProgramTest : public ::testing::Test {
  protected:
    void SetUp() override {
      CUDA_CHECK(cudaStreamCreate(&stream));
      handle.set_stream(stream);

      // Params
      hyper_params.population_size = 2;
      hyper_params.random_state = 123;
      hyper_params.num_features = 3;

      // X[0] * X[1] + X[2] + 0.5
      h_nodes1.push_back(node(node::type::add));
      h_nodes1.push_back(node(node::type::add));
      h_nodes1.push_back(node(node::type::mul));
      h_nodes1.push_back(node(0));
      h_nodes1.push_back(node(1));
      h_nodes1.push_back(node(2));
      h_nodes1.push_back(node(0.5f));

      // 0.5*X[1] - 0.4*X[2]
      h_nodes2.push_back(node(node::type::sub));
      h_nodes2.push_back(node(node::type::mul));
      h_nodes2.push_back(node(0.5f));
      h_nodes2.push_back(node(1));
      h_nodes2.push_back(node(node::type::mul));
      h_nodes2.push_back(node(0.4f));
      h_nodes2.push_back(node(2));
      
      // Programs
      h_progs.resize(2);
      h_progs[0].nodes = h_nodes1.data(); 
      h_progs[0].len = h_nodes1.size();

      h_progs[1].nodes = h_nodes2.data(); 
      h_progs[1].len = h_nodes1.size();

      // Device memory
      d_data  = (float*)handle.get_device_allocator()->allocate(75*sizeof(float),stream);
      d_y     = (float*)handle.get_device_allocator()->allocate(25*sizeof(float),stream);

      d_nodes1 = (node*)handle.get_device_allocator()->allocate(7*sizeof(node),stream);
      d_nodes2 = (node*)handle.get_device_allocator()->allocate(7*sizeof(node),stream);

      d_progs = (program_t)handle.get_device_allocator()->allocate(2*sizeof(program),stream);

      CUDA_CHECK(cudaMemcpyAsync(d_data,h_data.data(),75*sizeof(float),cudaMemcpyHostToDevice,stream));
      CUDA_CHECK(cudaMemcpyAsync(d_y, h_y.data(),25*sizeof(float),cudaMemcpyHostToDevice,stream));
      
      CUDA_CHECK(cudaMemcpyAsync(d_nodes1, h_nodes1.data(), 7*sizeof(node),cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_nodes2, h_nodes2.data(), 7*sizeof(node),cudaMemcpyHostToDevice, stream));

      program_t tmp = new program(h_progs[0],false);
      tmp->nodes = d_nodes1;
      CUDA_CHECK(cudaMemcpyAsync(&d_progs[0],tmp,sizeof(program),cudaMemcpyHostToDevice,stream));

      tmp = new program(h_progs[1],false);
      tmp->nodes = d_nodes2;
      CUDA_CHECK(cudaMemcpyAsync(&d_progs[1],tmp,sizeof(program),cudaMemcpyHostToDevice,stream));

    }

    void TearDown() override {
      CUDA_CHECK(cudaFree(d_nodes1));
      CUDA_CHECK(cudaFree(d_nodes2));
      CUDA_CHECK(cudaFree(d_progs));
      CUDA_CHECK(cudaFree(d_data));
      CUDA_CHECK(cudaFree(d_y));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }

    raft::handle_t handle;
    cudaStream_t stream;
    const int n_rows = 20;
    const int n_cols = 3;
    const int n_progs = 2;
    const int n_samples = 10;
    
    // 25*3 datapoints generated using scikit-learn
    // y = X[0] * X[1] + X[2] + 0.5
    std::vector<float> h_data { -0.50446586, -2.06014071,  0.88514116, -2.3015387 ,  0.83898341,
                                1.65980218, -0.87785842,  0.31563495,  0.3190391 ,  0.53035547,
                                0.30017032, -0.12289023, -1.10061918, -0.0126646 ,  2.10025514,
                                1.13376944, -0.88762896,  0.05080775, -0.34934272,  2.18557541,
                                0.50249434, -0.07557171, -0.52817175, -0.6871727 ,  0.51292982,
                                -1.44411381,  1.46210794,  0.28558733,  0.86540763,  0.58662319,
                                0.2344157 , -0.17242821,  0.87616892, -0.7612069 , -0.26788808,
                                0.61720311, -0.68372786,  0.58281521, -0.67124613,  0.19091548,
                                -0.38405435, -0.19183555,  1.6924546 , -1.1425182 ,  1.51981682,
                                0.90159072,  0.48851815, -0.61175641, -0.39675353,  1.25286816,
                                -1.39649634, -0.24937038,  0.93110208, -1.07296862, -0.20889423,
                                -1.11731035, -1.09989127,  0.16003707,  1.74481176, -0.93576943,
                                0.12015895,  0.90085595,  0.04221375, -0.84520564, -0.63699565,
                                -0.3224172 ,  0.74204416, -0.74715829, -0.35224985,  1.13162939,
                                1.14472371, -0.29809284,  1.62434536, -0.69166075, -0.75439794  };

    std::vector<float> h_y    { -0.16799022, -2.76151846,  1.68388718, -2.56473777,  0.78327289,
                                -0.22822666, -0.44852371,  0.9365866 ,  2.001957  , -0.57784534,
                                0.80542501,  1.48487942, -0.09924385, -0.33670458,  0.26397558,
                                -0.2578463 ,  1.41232295, -0.16116848,  0.54688057,  4.95330364,
                                2.09776794,  0.16498901,  2.44745782,  0.08097744,  0.3882355   };
    
    std::vector<node> h_nodes1;
    std::vector<node> h_nodes2;
    std::vector<program> h_progs;

    node* d_nodes1;
    node* d_nodes2;
    program_t d_progs;
    float* d_data;
    float* d_y;

    param hyper_params;
};

TEST_F(GeneticProgramTest, PearsonLoss){
  // Tolerance level of 1e-2
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,SpearmanLoss){
  // Tolerance level of 1e-2
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,MeanSquareLoss){
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,MeanAbsoluteLoss){
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,RMSLoss){  
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,LogLoss){
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,ProgramExecution){
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,ProgramFitnessScore){
  ASSERT_EQ(1,1);
}

TEST_F(GeneticProgramTest,ProgramFitnessSet){
  ASSERT_EQ(1,1);
}

} // namespace genetic
} // namespace cuml