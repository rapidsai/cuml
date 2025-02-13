/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <cuml/genetic/node.h>
#include <cuml/genetic/program.h>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace cuml {
namespace genetic {

class GeneticProgramTest : public ::testing::Test {
 public:
  GeneticProgramTest()
    : d_data(0, cudaStream_t(0)),
      d_y(0, cudaStream_t(0)),
      d_lYpred(0, cudaStream_t(0)),
      d_lY(0, cudaStream_t(0)),
      d_lunitW(0, cudaStream_t(0)),
      d_lW(0, cudaStream_t(0)),
      dx2(0, cudaStream_t(0)),
      dy2(0, cudaStream_t(0)),
      dw2(0, cudaStream_t(0)),
      dyp2(0, cudaStream_t(0)),
      stream(handle.get_stream())
  {
  }

 protected:
  void SetUp() override
  {
    // Params
    hyper_params.population_size = 2;
    hyper_params.random_state    = 123;
    hyper_params.num_features    = 3;

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
    h_progs[0].len   = h_nodes1.size();
    h_progs[0].nodes = new node[h_progs[0].len];
    std::copy(h_nodes1.data(), h_nodes1.data() + h_nodes1.size(), h_progs[0].nodes);

    h_progs[1].len   = h_nodes2.size();
    h_progs[1].nodes = new node[h_progs[1].len];
    std::copy(h_nodes2.data(), h_nodes2.data() + h_nodes2.size(), h_progs[1].nodes);

    // Loss weights
    h_lunitW.resize(250, 1.0f);

    // Smaller input
    hw2.resize(5, 1.0f);

    // Device memory
    d_data.resize(75, stream);
    d_y.resize(25, stream);
    d_lYpred.resize(500, stream);
    d_lY.resize(250, stream);
    d_lunitW.resize(250, stream);
    d_lW.resize(250, stream);
    d_nodes1 = (node*)rmm::mr::get_current_device_resource()->allocate(7 * sizeof(node), stream);
    d_nodes2 = (node*)rmm::mr::get_current_device_resource()->allocate(7 * sizeof(node), stream);
    d_progs =
      (program_t)rmm::mr::get_current_device_resource()->allocate(2 * sizeof(program), stream);

    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_lYpred.data(), h_lYpred.data(), 500 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_lY.data(), h_lY.data(), 250 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_lunitW.data(), h_lunitW.data(), 250 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_lW.data(), h_lW.data(), 250 * sizeof(float), cudaMemcpyHostToDevice, stream));

    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_data.data(), h_data.data(), 75 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_y.data(), h_y.data(), 25 * sizeof(float), cudaMemcpyHostToDevice, stream));

    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_nodes1, h_nodes1.data(), 7 * sizeof(node), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_nodes2, h_nodes2.data(), 7 * sizeof(node), cudaMemcpyHostToDevice, stream));

    program tmp(h_progs[0]);
    delete[] tmp.nodes;
    tmp.nodes = d_nodes1;
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(&d_progs[0], &tmp, sizeof(program), cudaMemcpyHostToDevice, stream));
    tmp.nodes = nullptr;

    tmp = program(h_progs[1]);
    delete[] tmp.nodes;
    tmp.nodes = d_nodes2;
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(&d_progs[1], &tmp, sizeof(program), cudaMemcpyHostToDevice, stream));
    tmp.nodes = nullptr;

    // Small input
    dx2.resize(15, stream);
    dy2.resize(5, stream);
    dw2.resize(5, stream);
    dyp2.resize(10, stream);

    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dx2.data(), hx2.data(), 15 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dy2.data(), hy2.data(), 5 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dw2.data(), hw2.data(), 5 * sizeof(float), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      dyp2.data(), hyp2.data(), 10 * sizeof(float), cudaMemcpyHostToDevice, stream));
  }

  void TearDown() override
  {
    rmm::mr::get_current_device_resource()->deallocate(d_nodes1, 7 * sizeof(node), stream);
    rmm::mr::get_current_device_resource()->deallocate(d_nodes2, 7 * sizeof(node), stream);
    rmm::mr::get_current_device_resource()->deallocate(d_progs, 2 * sizeof(program), stream);
  }

  raft::handle_t handle;
  cudaStream_t stream;
  const int n_cols      = 3;
  const int n_progs     = 2;
  const int n_samples   = 25;
  const int n_samples2  = 5;
  const float tolerance = 0.025f;  // assuming up to 2.5% tolerance for results(for now)

  // 25*3 datapoints generated using numpy
  // y = X[0] * X[1] + X[2] + 0.5
  std::vector<float> h_data{
    -0.50446586, -2.06014071, 0.88514116,  -2.3015387,  0.83898341,  1.65980218,  -0.87785842,
    0.31563495,  0.3190391,   0.53035547,  0.30017032,  -0.12289023, -1.10061918, -0.0126646,
    2.10025514,  1.13376944,  -0.88762896, 0.05080775,  -0.34934272, 2.18557541,  0.50249434,
    -0.07557171, -0.52817175, -0.6871727,  0.51292982,

    -1.44411381, 1.46210794,  0.28558733,  0.86540763,  0.58662319,  0.2344157,   -0.17242821,
    0.87616892,  -0.7612069,  -0.26788808, 0.61720311,  -0.68372786, 0.58281521,  -0.67124613,
    0.19091548,  -0.38405435, -0.19183555, 1.6924546,   -1.1425182,  1.51981682,  0.90159072,
    0.48851815,  -0.61175641, -0.39675353, 1.25286816,

    -1.39649634, -0.24937038, 0.93110208,  -1.07296862, -0.20889423, -1.11731035, -1.09989127,
    0.16003707,  1.74481176,  -0.93576943, 0.12015895,  0.90085595,  0.04221375,  -0.84520564,
    -0.63699565, -0.3224172,  0.74204416,  -0.74715829, -0.35224985, 1.13162939,  1.14472371,
    -0.29809284, 1.62434536,  -0.69166075, -0.75439794};

  std::vector<float> h_y{-0.16799022, -2.76151846, 1.68388718,  -2.56473777, 0.78327289,
                         -0.22822666, -0.44852371, 0.9365866,   2.001957,    -0.57784534,
                         0.80542501,  1.48487942,  -0.09924385, -0.33670458, 0.26397558,
                         -0.2578463,  1.41232295,  -0.16116848, 0.54688057,  4.95330364,
                         2.09776794,  0.16498901,  2.44745782,  0.08097744,  0.3882355};

  // Values for loss function tests (250 values each)
  std::vector<float> h_lYpred{
    0.06298f, 0.81894f, 0.12176f, 0.17104f, 0.12851f, 0.28721f, 0.85043f, 0.68120f, 0.57074f,
    0.21796f, 0.96626f, 0.32337f, 0.21887f, 0.80867f, 0.96438f, 0.20052f, 0.28668f, 0.86931f,
    0.71421f, 0.85405f, 0.13916f, 0.00316f, 0.59440f, 0.86299f, 0.67019f, 0.54309f, 0.82629f,
    0.94563f, 0.01481f, 0.13665f, 0.77081f, 0.58024f, 0.02538f, 0.36610f, 0.13948f, 0.75034f,
    0.80435f, 0.27488f, 0.74165f, 0.02921f, 0.51479f, 0.66415f, 0.27380f, 0.85304f, 0.95767f,
    0.22758f, 0.38602f, 0.41555f, 0.53783f, 0.48663f, 0.11103f, 0.69397f, 0.21749f, 0.71930f,
    0.28976f, 0.50971f, 0.68532f, 0.97518f, 0.71299f, 0.37629f, 0.56444f, 0.42280f, 0.51921f,
    0.84366f, 0.30778f, 0.39493f, 0.74007f, 0.18280f, 0.22621f, 0.63083f, 0.46085f, 0.47259f,
    0.65442f, 0.25453f, 0.23058f, 0.17460f, 0.30702f, 0.22421f, 0.37237f, 0.36660f, 0.29702f,
    0.65276f, 0.30222f, 0.63844f, 0.99909f, 0.55084f, 0.05066f, 0.18914f, 0.36652f, 0.36765f,
    0.93901f, 0.13575f, 0.72582f, 0.20223f, 0.06375f, 0.52581f, 0.77119f, 0.12127f, 0.27800f,
    0.04008f, 0.01752f, 0.00394f, 0.68973f, 0.91931f, 0.48011f, 0.48363f, 0.09770f, 0.84381f,
    0.80244f, 0.42710f, 0.82164f, 0.63239f, 0.08117f, 0.46195f, 0.49832f, 0.05717f, 0.16886f,
    0.22311f, 0.45326f, 0.50748f, 0.19089f, 0.78211f, 0.34272f, 0.38456f, 0.64874f, 0.18216f,
    0.64757f, 0.26900f, 0.20780f, 0.87067f, 0.16903f, 0.77285f, 0.70580f, 0.54404f, 0.97395f,
    0.52550f, 0.81364f, 0.30085f, 0.36754f, 0.42492f, 0.79470f, 0.31590f, 0.26322f, 0.68332f,
    0.96523f, 0.31110f, 0.97029f, 0.80217f, 0.77125f, 0.36302f, 0.13444f, 0.28420f, 0.20442f,
    0.89692f, 0.50515f, 0.61952f, 0.48237f, 0.35080f, 0.75606f, 0.85438f, 0.70647f, 0.91793f,
    0.24037f, 0.72867f, 0.84713f, 0.39838f, 0.49553f, 0.32876f, 0.22610f, 0.86573f, 0.99232f,
    0.71321f, 0.30179f, 0.01941f, 0.84838f, 0.58587f, 0.43339f, 0.29490f, 0.07191f, 0.88531f,
    0.26896f, 0.36085f, 0.96043f, 0.70679f, 0.39593f, 0.37642f, 0.76078f, 0.63827f, 0.36346f,
    0.12755f, 0.07074f, 0.67744f, 0.35042f, 0.30773f, 0.15577f, 0.64096f, 0.05035f, 0.32882f,
    0.33640f, 0.54106f, 0.76279f, 0.00414f, 0.17373f, 0.83551f, 0.18176f, 0.91190f, 0.03559f,
    0.31992f, 0.86311f, 0.04054f, 0.49714f, 0.53551f, 0.65316f, 0.15681f, 0.80268f, 0.44978f,
    0.26365f, 0.37162f, 0.97630f, 0.82863f, 0.73267f, 0.93207f, 0.47129f, 0.70817f, 0.57300f,
    0.34240f, 0.89749f, 0.79844f, 0.67992f, 0.72523f, 0.43319f, 0.07310f, 0.61074f, 0.93830f,
    0.90822f, 0.08077f, 0.28048f, 0.04549f, 0.44870f, 0.10337f, 0.93911f, 0.13464f, 0.16080f,
    0.94620f, 0.15276f, 0.56239f, 0.38684f, 0.12437f, 0.98149f, 0.80650f, 0.44040f, 0.59698f,
    0.82197f, 0.91634f, 0.89667f, 0.96333f, 0.21204f, 0.47457f, 0.95737f, 0.08697f, 0.50921f,
    0.58647f, 0.71985f, 0.39455f, 0.73240f, 0.04227f, 0.74879f, 0.34403f, 0.94240f, 0.45158f,
    0.83860f, 0.51819f, 0.87374f, 0.70416f, 0.52987f, 0.72727f, 0.53649f, 0.74878f, 0.13247f,
    0.91358f, 0.61871f, 0.50048f, 0.04681f, 0.56370f, 0.68393f, 0.51947f, 0.85044f, 0.24416f,
    0.39354f, 0.33526f, 0.66574f, 0.65638f, 0.15506f, 0.84167f, 0.84663f, 0.92094f, 0.14140f,
    0.69364f, 0.40575f, 0.63543f, 0.35074f, 0.68887f, 0.70662f, 0.90424f, 0.09042f, 0.57486f,
    0.52239f, 0.40711f, 0.82103f, 0.08674f, 0.14005f, 0.44922f, 0.81244f, 0.99037f, 0.26577f,
    0.64744f, 0.25391f, 0.47913f, 0.09676f, 0.26023f, 0.86098f, 0.24472f, 0.15364f, 0.38980f,
    0.02943f, 0.59390f, 0.25683f, 0.38976f, 0.90195f, 0.27418f, 0.45255f, 0.74992f, 0.07155f,
    0.95425f, 0.77560f, 0.41618f, 0.27963f, 0.32602f, 0.75690f, 0.09356f, 0.73795f, 0.59604f,
    0.97534f, 0.27677f, 0.06770f, 0.59517f, 0.64286f, 0.36224f, 0.22017f, 0.83546f, 0.21461f,
    0.24793f, 0.08248f, 0.16668f, 0.74429f, 0.66674f, 0.68034f, 0.34710f, 0.82358f, 0.47555f,
    0.50109f, 0.09328f, 0.98566f, 0.99481f, 0.41391f, 0.86833f, 0.38645f, 0.49203f, 0.44547f,
    0.55391f, 0.87598f, 0.85542f, 0.56283f, 0.61385f, 0.70564f, 0.29067f, 0.91150f, 0.64787f,
    0.18255f, 0.03792f, 0.69633f, 0.29029f, 0.31412f, 0.49111f, 0.34615f, 0.43144f, 0.31616f,
    0.15405f, 0.44915f, 0.12777f, 0.09491f, 0.26003f, 0.71537f, 0.19450f, 0.91570f, 0.28420f,
    0.77892f, 0.53199f, 0.66034f, 0.01978f, 0.35415f, 0.03664f, 0.42675f, 0.41304f, 0.33804f,
    0.11290f, 0.89985f, 0.75959f, 0.59417f, 0.53113f, 0.38898f, 0.76259f, 0.83973f, 0.75809f,
    0.65900f, 0.55141f, 0.14175f, 0.44740f, 0.95823f, 0.77612f, 0.48749f, 0.74491f, 0.57491f,
    0.59119f, 0.26665f, 0.48599f, 0.85947f, 0.46245f, 0.08129f, 0.00825f, 0.29669f, 0.43499f,
    0.47998f, 0.60173f, 0.26611f, 0.01223f, 0.81734f, 0.77892f, 0.79022f, 0.01394f, 0.45596f,
    0.45259f, 0.32536f, 0.84229f, 0.43612f, 0.30531f, 0.10670f, 0.57758f, 0.65956f, 0.42007f,
    0.32166f, 0.10552f, 0.63558f, 0.17990f, 0.50732f, 0.34599f, 0.16603f, 0.26309f, 0.04098f,
    0.15997f, 0.79728f, 0.00528f, 0.35510f, 0.24344f, 0.07018f, 0.22062f, 0.92927f, 0.13373f,
    0.50955f, 0.11199f, 0.75728f, 0.62117f, 0.18153f, 0.84993f, 0.04677f, 0.13013f, 0.92211f,
    0.95474f, 0.88898f, 0.55561f, 0.22625f, 0.78700f, 0.73659f, 0.97613f, 0.02299f, 0.07724f,
    0.78942f, 0.02193f, 0.05320f, 0.92053f, 0.35103f, 0.39305f, 0.24208f, 0.08225f, 0.78460f,
    0.52144f, 0.32927f, 0.84725f, 0.36106f, 0.80349f};

  std::vector<float> h_lY{
    0.60960f, 0.61090f, 0.41418f, 0.90827f, 0.76181f, 0.31777f, 0.04096f, 0.27290f, 0.56879f,
    0.75461f, 0.73555f, 0.41598f, 0.59506f, 0.08768f, 0.99554f, 0.20613f, 0.13546f, 0.32044f,
    0.41057f, 0.38501f, 0.27894f, 0.24027f, 0.91171f, 0.26811f, 0.55595f, 0.71153f, 0.69739f,
    0.53411f, 0.78365f, 0.60914f, 0.41856f, 0.61688f, 0.28741f, 0.28708f, 0.37029f, 0.47945f,
    0.40612f, 0.75762f, 0.91728f, 0.70406f, 0.26717f, 0.71175f, 0.39243f, 0.35904f, 0.38469f,
    0.08664f, 0.38611f, 0.35606f, 0.52801f, 0.96986f, 0.84780f, 0.56942f, 0.41712f, 0.17005f,
    0.79105f, 0.74347f, 0.83473f, 0.06303f, 0.37864f, 0.66666f, 0.78153f, 0.11061f, 0.33880f,
    0.82412f, 0.47141f, 0.53043f, 0.51184f, 0.34172f, 0.57087f, 0.88349f, 0.32870f, 0.11501f,
    0.35460f, 0.23630f, 0.37728f, 0.96120f, 0.19871f, 0.78119f, 0.23860f, 0.70615f, 0.46745f,
    0.43392f, 0.49967f, 0.39721f, 0.53185f, 0.27827f, 0.14435f, 0.82008f, 0.43275f, 0.82113f,
    0.06428f, 0.53528f, 0.21594f, 0.86172f, 0.41172f, 0.96051f, 0.54487f, 0.01971f, 0.71222f,
    0.04258f, 0.36715f, 0.24844f, 0.12494f, 0.34132f, 0.87059f, 0.70216f, 0.33533f, 0.10020f,
    0.79337f, 0.26059f, 0.81314f, 0.54342f, 0.79115f, 0.71730f, 0.70860f, 0.00998f, 0.64761f,
    0.01206f, 0.53463f, 0.94436f, 0.19639f, 0.23296f, 0.55945f, 0.14070f, 0.57765f, 0.50908f,
    0.95720f, 0.95611f, 0.12311f, 0.95382f, 0.23116f, 0.36939f, 0.66395f, 0.76282f, 0.16314f,
    0.00186f, 0.77662f, 0.58799f, 0.18155f, 0.10355f, 0.45982f, 0.34359f, 0.59476f, 0.72759f,
    0.77310f, 0.50736f, 0.43720f, 0.63624f, 0.84569f, 0.73073f, 0.04179f, 0.64806f, 0.19924f,
    0.96082f, 0.06270f, 0.27744f, 0.59384f, 0.07317f, 0.10979f, 0.47857f, 0.60274f, 0.54937f,
    0.58563f, 0.45247f, 0.84396f, 0.43945f, 0.47719f, 0.40808f, 0.81152f, 0.48558f, 0.21577f,
    0.93935f, 0.08222f, 0.43114f, 0.68239f, 0.78870f, 0.24300f, 0.84829f, 0.44764f, 0.57347f,
    0.78353f, 0.30614f, 0.39493f, 0.40320f, 0.72849f, 0.39406f, 0.89363f, 0.33323f, 0.38395f,
    0.94783f, 0.46082f, 0.30498f, 0.17110f, 0.14083f, 0.48474f, 0.45024f, 0.92586f, 0.77450f,
    0.43503f, 0.45188f, 0.80866f, 0.24937f, 0.34205f, 0.35942f, 0.79689f, 0.77224f, 0.14354f,
    0.54387f, 0.50787f, 0.31753f, 0.98414f, 0.03261f, 0.89748f, 0.82350f, 0.60235f, 0.00041f,
    0.99696f, 0.39894f, 0.52078f, 0.54421f, 0.33405f, 0.81143f, 0.49764f, 0.44993f, 0.37257f,
    0.16238f, 0.81337f, 0.51335f, 0.96118f, 0.98901f, 0.95259f, 0.36557f, 0.24654f, 0.99554f,
    0.33408f, 0.01734f, 0.85852f, 0.41286f, 0.67371f, 0.93781f, 0.04977f, 0.17298f, 0.91502f,
    0.70144f, 0.97356f, 0.12571f, 0.64375f, 0.10033f, 0.36798f, 0.90001f};

  // Unitary weights
  std::vector<float> h_lunitW;

  // Non-unitary weights
  std::vector<float> h_lW{
    0.38674f, 0.59870f, 0.36761f, 0.59731f, 0.99057f, 0.24131f, 0.29727f, 0.94112f, 0.78962f,
    0.71998f, 0.10983f, 0.33620f, 0.37988f, 0.14344f, 0.37377f, 0.06403f, 0.22877f, 0.21993f,
    0.11340f, 0.28554f, 0.45453f, 0.14344f, 0.11715f, 0.23184f, 0.08622f, 0.26746f, 0.49058f,
    0.06981f, 0.41885f, 0.04422f, 0.99925f, 0.71709f, 0.11910f, 0.49944f, 0.98116f, 0.66316f,
    0.11646f, 0.25202f, 0.93223f, 0.81414f, 0.20446f, 0.23813f, 0.45380f, 0.83618f, 0.95958f,
    0.72684f, 0.86808f, 0.96348f, 0.76092f, 0.86071f, 0.44155f, 0.85212f, 0.76185f, 0.51460f,
    0.65627f, 0.38269f, 0.08251f, 0.07506f, 0.22281f, 0.05325f, 0.71190f, 0.62834f, 0.19348f,
    0.44271f, 0.23677f, 0.81817f, 0.73055f, 0.48816f, 0.57524f, 0.45278f, 0.27998f, 0.35699f,
    0.26875f, 0.63546f, 0.50990f, 0.21046f, 0.76892f, 0.74433f, 0.39302f, 0.55071f, 0.24554f,
    0.56793f, 0.67852f, 0.43290f, 0.97266f, 0.52475f, 0.88402f, 0.79439f, 0.01496f, 0.46426f,
    0.15537f, 0.35364f, 0.42962f, 0.47999f, 0.06357f, 0.78531f, 0.62165f, 0.45226f, 0.84973f,
    0.63747f, 0.00593f, 0.31520f, 0.13150f, 0.47776f, 0.56420f, 0.21679f, 0.32107f, 0.62491f,
    0.33747f, 0.86599f, 0.82573f, 0.26970f, 0.50087f, 0.86947f, 0.47433f, 0.91848f, 0.19534f,
    0.45760f, 0.38407f, 0.18953f, 0.30000f, 0.37964f, 0.42509f, 0.55408f, 0.74500f, 0.44484f,
    0.67679f, 0.12214f, 0.68380f, 0.74917f, 0.87429f, 0.04355f, 0.98426f, 0.88845f, 0.88318f,
    0.64393f, 0.90849f, 0.87948f, 0.22915f, 0.86887f, 0.58676f, 0.51575f, 0.56549f, 0.41412f,
    0.06593f, 0.40484f, 0.72931f, 0.02289f, 0.96391f, 0.61075f, 0.91701f, 0.29698f, 0.37095f,
    0.42087f, 0.73251f, 0.93271f, 0.32687f, 0.48981f, 0.01081f, 0.11985f, 0.46962f, 0.02569f,
    0.83989f, 0.21767f, 0.82370f, 0.35174f, 0.94939f, 0.46032f, 0.81569f, 0.66635f, 0.07019f,
    0.68926f, 0.65628f, 0.19914f, 0.17936f, 0.64540f, 0.09031f, 0.05875f, 0.88790f, 0.83687f,
    0.46605f, 0.08537f, 0.49514f, 0.44504f, 0.67687f, 0.28943f, 0.74668f, 0.43207f, 0.70990f,
    0.62513f, 0.56137f, 0.94399f, 0.75806f, 0.41840f, 0.38428f, 0.30754f, 0.62633f, 0.23173f,
    0.40750f, 0.49968f, 0.05536f, 0.11405f, 0.34185f, 0.36367f, 0.06341f, 0.66834f, 0.42899f,
    0.08343f, 0.72266f, 0.33155f, 0.74943f, 0.15387f, 0.02475f, 0.35741f, 0.15806f, 0.35406f,
    0.18226f, 0.31042f, 0.36047f, 0.62366f, 0.30036f, 0.66625f, 0.99695f, 0.99472f, 0.06743f,
    0.56804f, 0.28185f, 0.77387f, 0.58763f, 0.77824f, 0.03720f, 0.99490f, 0.73720f, 0.93635f,
    0.85669f, 0.91634f, 0.26065f, 0.97469f, 0.03867f, 0.52306f, 0.99167f, 0.90332f, 0.88546f,
    0.07109f, 0.94168f, 0.10211f, 0.95949f, 0.86314f, 0.59917f, 0.41948f};

  // Setup smaller input
  std::vector<float> hx2 = {0.06298,
                            0.96626,
                            0.13916,
                            0.77081,
                            0.51479,
                            0.81894,
                            0.32337,
                            0.00316,
                            0.58024,
                            0.66415,
                            0.12176,
                            0.21887,
                            0.59440,
                            0.02538,
                            0.27380};

  std::vector<float> hy2  = {0.11103, 0.69397, 0.21749, 0.71930, 0.28976};
  std::vector<float> hyp2 = {
    0.67334, 1.03133, 1.09484, 0.97263, 1.1157, 0.36077, 0.07413, -0.23618, 0.27997, 0.22255};
  std::vector<float> hw2;

  // Nodes and programs
  std::vector<node> h_nodes1;
  std::vector<node> h_nodes2;
  std::vector<program> h_progs;

  // Device ptrs
  node* d_nodes1;
  node* d_nodes2;
  program_t d_progs;
  rmm::device_uvector<float> d_data;
  rmm::device_uvector<float> d_y;
  rmm::device_uvector<float> d_lYpred;
  rmm::device_uvector<float> d_lY;
  rmm::device_uvector<float> d_lunitW;
  rmm::device_uvector<float> d_lW;
  rmm::device_uvector<float> dx2;
  rmm::device_uvector<float> dy2;
  rmm::device_uvector<float> dw2;
  rmm::device_uvector<float> dyp2;

  param hyper_params;
};

TEST_F(GeneticProgramTest, PearsonCoeff)
{
  MLCommon::CompareApproxAbs<float> compApprox(tolerance);
  float h_expected_score[2] = {0.09528403f, 0.08269963f};
  float h_score[2]          = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::pearson;

  // Unitary weights
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Unitary weights - small
  h_expected_score[0] = 0.3247632f;
  h_expected_score[1] = 0.0796348f;
  compute_metric(
    handle, n_samples2, n_progs, dy2.data(), dyp2.data(), dw2.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights
  h_expected_score[0] = 0.14329584f;
  h_expected_score[1] = 0.09064283f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, SpearmanCoeff)
{
  MLCommon::CompareApproxAbs<float> compApprox(tolerance);
  float h_score[2] = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::spearman;

  // Unitary weights
  float h_expected_score[2] = {0.09268333f, 0.07529861f};
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Unitary weights - small
  h_expected_score[0] = 0.10000f;
  h_expected_score[1] = 0.10000f;
  compute_metric(
    handle, n_samples2, n_progs, dy2.data(), dyp2.data(), dw2.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights
  h_expected_score[0] = 0.14072408f;
  h_expected_score[1] = 0.08157397f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, MeanSquareLoss)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);
  float h_score[2] = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::mse;

  // Unitary weights
  float h_expected_score[2] = {0.14297023, 0.14242104};
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Unitary weights - small
  h_expected_score[0] = 0.3892163f;
  h_expected_score[1] = 0.1699830f;
  compute_metric(
    handle, n_samples2, n_progs, dy2.data(), dyp2.data(), dw2.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights
  h_expected_score[0] = 0.13842479f;
  h_expected_score[1] = 0.14538825f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, MeanAbsoluteLoss)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);
  float h_score[2] = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::mae;

  // Unitary weights - big
  float h_expected_score[2] = {0.30614017, 0.31275677};
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Unitary weights - small
  h_expected_score[0] = 0.571255f;
  h_expected_score[1] = 0.365957f;
  compute_metric(
    handle, n_samples2, n_progs, dy2.data(), dyp2.data(), dw2.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights -big
  h_expected_score[0] = 0.29643119f;
  h_expected_score[1] = 0.31756123f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, RMSLoss)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);
  float h_score[2] = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::rmse;

  // Unitary weights
  float h_expected_score[2] = {0.37811404, 0.37738713};
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Unitary weights - small
  h_expected_score[0] = 0.6238720f;
  h_expected_score[1] = 0.4122899f;
  compute_metric(
    handle, n_samples2, n_progs, dy2.data(), dyp2.data(), dw2.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights
  h_expected_score[0] = 0.37205482f;
  h_expected_score[1] = 0.38129811f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, LogLoss)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);
  float h_score[2] = {0.0f, 0.0f};
  rmm::device_uvector<float> d_score(2, stream);
  hyper_params.metric = metric_t::logloss;

  // Unitary weights
  float h_expected_score[2] = {0.72276, 0.724011};
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lunitW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }

  // Non-unitary weights
  h_expected_score[0] = 0.715887f;
  h_expected_score[1] = 0.721293f;
  compute_metric(
    handle, 250, 2, d_lY.data(), d_lYpred.data(), d_lW.data(), d_score.data(), hyper_params);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(h_score, d_score.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  std::copy(h_score, h_score + 2, std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(compApprox(h_score[i], h_expected_score[i]));
  }
}

TEST_F(GeneticProgramTest, ProgramExecution)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);

  // Enable debug logging
  ML::default_logger().set_level(rapids_logger::level_enum::info);

  // Allocate memory
  std::vector<float> h_ypred(n_progs * n_samples, 0.0f);
  rmm::device_uvector<float> d_ypred(n_progs * n_samples, stream);

  // Execute programs
  execute(handle, d_progs, n_samples, n_progs, d_data.data(), d_ypred.data());
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_ypred.data(),
                                d_ypred.data(),
                                n_progs * n_samples * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream));
  handle.sync_stream(stream);

  // Check results

  for (int i = 0; i < n_samples; ++i) {
    ASSERT_TRUE(compApprox(h_ypred[i], h_y[i]));
  }

  for (int i = 0; i < n_samples; ++i) {
    ASSERT_TRUE(compApprox(h_ypred[n_samples + i],
                           0.5 * h_data[n_samples + i] - 0.4 * h_data[2 * n_samples + i]));
  }
}

TEST_F(GeneticProgramTest, ProgramFitnessScore)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);

  std::vector<metric_t> all_metrics = {
    metric_t::mae, metric_t::mse, metric_t::rmse, metric_t::pearson, metric_t::spearman};

  std::vector<float> hexpscores = {
    0.57126, 0.36596, 0.38922, 0.16998, 0.62387, 0.41229, 0.32476, 0.07963, 0.10000, 0.10000};

  std::vector<float> hactualscores(10);

  rmm::device_uvector<float> dactualscores(10, stream);

  // Start execution for all metrics
  for (int i = 0; i < 5; ++i) {
    hyper_params.metric = all_metrics[i];
    find_batched_fitness(handle,
                         n_progs,
                         d_progs,
                         dactualscores.data() + 2 * i,
                         hyper_params,
                         n_samples2,
                         dx2.data(),
                         dy2.data(),
                         dw2.data());
    handle.sync_stream(stream);
  }

  RAFT_CUDA_TRY(cudaMemcpyAsync(hactualscores.data(),
                                dactualscores.data(),
                                10 * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream));
  std::copy(
    hactualscores.begin(), hactualscores.end(), std::ostream_iterator<float>(std::cerr, ";"));
  std::cerr << std::endl;

  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(compApprox(std::abs(hactualscores[i]), hexpscores[i]));
  }
}

}  // namespace genetic
}  // namespace cuml
