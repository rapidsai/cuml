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
#include <cuml/genetic/genetic.h>
#include <cuml/genetic/node.h>
#include <cuml/genetic/program.h>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace cuml {
namespace genetic {

/**
 * @brief Tests the training and inference of the symbolic regressor, classifier and transformer
 *        on y = 0.5X[0] + 0.4 X[1]
 *
 */
class GeneticEvolutionTest : public ::testing::Test {
 public:
  GeneticEvolutionTest()
    : d_train(0, cudaStream_t(0)),
      d_trainlab(0, cudaStream_t(0)),
      d_test(0, cudaStream_t(0)),
      d_testlab(0, cudaStream_t(0)),
      d_trainwts(0, cudaStream_t(0)),
      d_testwts(0, cudaStream_t(0)),
      stream(handle.get_stream())
  {
  }

 protected:
  void SetUp() override
  {
    ML::default_logger().set_level(rapids_logger::level_enum::info);

    // Set training param vals
    hyper_params.population_size       = 5000;
    hyper_params.num_features          = n_cols;
    hyper_params.random_state          = 11;
    hyper_params.generations           = 20;
    hyper_params.stopping_criteria     = 0.01;
    hyper_params.p_crossover           = 0.7;
    hyper_params.p_subtree_mutation    = 0.1;
    hyper_params.p_hoist_mutation      = 0.05;
    hyper_params.p_point_mutation      = 0.1;
    hyper_params.parsimony_coefficient = 0.01;

    // Initialize weights
    h_trainwts.resize(n_tr_rows, 1.0f);
    h_testwts.resize(n_tst_rows, 1.0f);

    // resize device memory
    d_train.resize(n_cols * n_tr_rows, stream);
    d_trainlab.resize(n_tr_rows, stream);
    d_test.resize(n_cols * n_tst_rows, stream);
    d_testlab.resize(n_tst_rows, stream);
    d_trainwts.resize(n_tr_rows, stream);
    d_testwts.resize(n_tst_rows, stream);

    // Memcpy HtoD
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_train.data(),
                                  h_train.data(),
                                  n_cols * n_tr_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_trainlab.data(),
                                  h_trainlab.data(),
                                  n_tr_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_test.data(),
                                  h_test.data(),
                                  n_cols * n_tst_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_testlab.data(),
                                  h_testlab.data(),
                                  n_tst_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_trainwts.data(),
                                  h_trainwts.data(),
                                  n_tr_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_testwts.data(),
                                  h_testwts.data(),
                                  n_tst_rows * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
  }

  raft::handle_t handle;
  cudaStream_t stream;
  param hyper_params;

  // Some mini-dataset constants
  const int n_tr_rows   = 250;
  const int n_tst_rows  = 50;
  const int n_cols      = 2;
  const float tolerance = 0.025f;  // assuming up to 2.5% tolerance for results(for now)

  // Contains synthetic Data
  // y =
  std::vector<float> h_train = {
    0.2119566,  -0.7221057, 0.9944866,  -0.6420138, 0.3243210,  -0.8062112, 0.9247920,  -0.8267401,
    0.2330494,  0.1486086,  -0.0957095, 0.1386102,  0.1674080,  0.0356288,  0.4644501,  0.3442579,
    0.6560287,  0.2349779,  -0.3978628, 0.1793082,  -0.1155355, 0.0176618,  0.8318791,  0.7813108,
    0.2736598,  0.6475824,  -0.3849131, -0.4696701, -0.6907704, 0.2952283,  -0.8723270, -0.3355115,
    -0.0523054, -0.8182662, 0.5539537,  -0.8737933, 0.5849895,  -0.2579604, 0.3574578,  -0.1654855,
    -0.2554073, 0.3591112,  0.9403976,  -0.3390219, 0.6517981,  0.6465558,  0.4370021,  -0.0079799,
    0.2970910,  0.2452746,  -0.7523201, -0.0951637, 0.6400041,  -0.5386036, 0.4352954,  -0.2126355,
    0.6203773,  0.7159789,  -0.6823127, 0.4670905,  -0.4666402, 0.0071169,  0.5038485,  -0.5780727,
    0.7944591,  0.6328644,  0.1813934,  0.2653100,  -0.1671608, 0.8108285,  0.3609906,  -0.5820257,
    0.0447571,  0.7247062,  0.3546630,  0.5908147,  -0.1850210, 0.8889677,  0.4725176,  0.2190818,
    0.1944676,  -0.1650774, 0.5239485,  0.4871244,  0.8803309,  0.3119077,  -0.1502819, 0.2140640,
    -0.3925484, 0.1745171,  -0.0332719, 0.9880465,  0.5828160,  0.3987538,  0.4770127,  -0.4151363,
    -0.9899210, 0.7880531,  -0.3253276, -0.4564783, -0.9825586, -0.0729553, 0.7512086,  0.3045725,
    -0.5038860, -0.9412159, -0.8188231, -0.3728235, 0.2280060,  -0.4212141, -0.2424457, -0.5574245,
    -0.5845115, 0.7049432,  -0.5244312, -0.0405502, -0.2238990, 0.6347900,  0.9998363,  0.3580613,
    0.0199144,  -0.1971139, 0.8036406,  0.7131155,  0.5613965,  0.3835140,  0.0717551,  0.0463067,
    0.5255786,  0.0928743,  0.1386557,  -0.7212757, 0.3051646,  0.2635859,  -0.5229289, -0.8547997,
    0.6653103,  -0.1116264, 0.2930650,  0.5135837,  0.7412015,  -0.3735900, -0.9826624, -0.6185324,
    -0.8464018, -0.4180478, 0.7254488,  -0.5188612, -0.3333993, 0.8999060,  -0.6015426, -0.6545046,
    0.6795465,  -0.5157862, 0.4536161,  -0.7564244, -0.0614987, 0.9840064,  0.3975551,  0.8684530,
    0.6091788,  0.2544823,  -0.9745569, -0.1815226, -0.1521985, 0.8436312,  -0.9446849, -0.2546227,
    0.9108996,  -0.2374187, -0.8820541, -0.2937101, 0.2558129,  0.7706293,  0.1066034,  -0.7223888,
    -0.6807924, -0.5187497, -0.3461997, 0.3319379,  -0.5073046, 0.0713026,  0.4598049,  -0.9708425,
    -0.2323956, 0.3963093,  -0.9132538, -0.2047350, 0.1162403,  -0.6301352, -0.1114944, -0.4411873,
    -0.7517651, 0.9942231,  0.6387486,  -0.3516690, 0.2925287,  0.8415794,  -0.2203800, 0.1182607,
    -0.5032156, 0.4939238,  0.9852490,  -0.8617036, -0.8945347, 0.1789286,  -0.1909516, 0.2587640,
    -0.2992706, 0.6049703,  -0.1238372, 0.8297717,  -0.3196876, 0.9792059,  0.7898732,  0.8210509,
    -0.5545098, -0.5691904, -0.7678227, -0.9643255, -0.1002291, -0.4273028, -0.6697328, -0.3049299,
    -0.0368014, 0.4804423,  -0.6646156, 0.5903011,  -0.1700153, -0.6397213, 0.9845422,  -0.5159376,
    0.1589690,  -0.3279489, -0.1498093, -0.9002322, 0.1960990,  0.3850992,  0.4812583,  -0.1506606,
    -0.0863564, -0.4061224, -0.3599582, -0.2919797, -0.5094189, 0.7824159,  0.3322580,  -0.3275573,
    -0.9909980, -0.5806390, 0.4667387,  -0.3746538, -0.7436752, 0.5058509,  0.5686203,  -0.8828574,
    0.2331149,  0.1225447,  0.9276860,  -0.2576783, -0.5962995, -0.6098081, -0.0473731, 0.6461973,
    -0.8618875, 0.2869696,  -0.5910612, 0.2354020,  0.7434812,  0.9635402,  -0.7473646, -0.1364276,
    0.4180313,  0.1777712,  -0.3155821, -0.3896985, -0.5973547, 0.3018475,  -0.2226010, 0.6965982,
    -0.1711176, 0.4426420,  0.5972827,  0.7491136,  0.5431328,  0.1888770,  -0.4517326, 0.7062291,
    0.5087549,  -0.3582025, -0.4492956, 0.1632529,  -0.1689859, 0.9334283,  -0.3891996, 0.1138209,
    0.7598738,  0.0241726,  -0.3133468, -0.0708007, 0.9602417,  -0.7650007, -0.6497396, 0.4096349,
    -0.7035034, 0.6052362,  0.5920056,  -0.4065195, 0.3722862,  -0.7039886, -0.2351859, 0.3143256,
    -0.8650362, 0.3481469,  0.5242298,  0.2190642,  0.7090682,  0.7368234,  0.3148258,  -0.8396302,
    -0.8332214, 0.6766308,  0.4428585,  0.5376374,  0.1104256,  -0.9560977, 0.8913012,  0.2302127,
    -0.7445556, -0.8753514, -0.1434969, 0.7423451,  -0.9627953, 0.7919458,  -0.8590292, -0.2405730,
    0.0733800,  -0.1964383, 0.3429065,  -0.5199867, -0.6148949, -0.4645573, -0.1036227, 0.1915514,
    0.4981042,  -0.3142545, -0.1360139, 0.5123143,  -0.8319357, 0.2593685,  -0.6637208, 0.8695423,
    -0.4745009, -0.4598881, 0.2561057,  0.8682946,  0.7572707,  -0.2405597, -0.6909520, -0.2329739,
    -0.3544887, 0.5916605,  -0.5483196, 0.3634111,  0.0485800,  0.1492287,  -0.0361141, 0.6510856,
    0.9754849,  -0.1871928, 0.7787021,  -0.6019276, 0.2416331,  -0.1160285, 0.8894659,  0.9423820,
    -0.7052383, -0.8790381, -0.7129928, 0.5332075,  -0.5728216, -0.9184565, 0.0437820,  0.3580015,
    -0.7459742, -0.6401960, -0.7465842, -0.0257084, 0.7586666,  0.3472861,  0.3226733,  -0.8356623,
    0.9038333,  0.9519323,  0.6794367,  -0.4118270, -0.1475553, 0.1638173,  0.7039975,  0.0782125,
    -0.6468386, -0.4905404, -0.0657285, -0.9094056, -0.1691999, 0.9545628,  0.5260556,  0.0704832,
    0.9559255,  0.4109315,  0.0437353,  0.1975988,  -0.2173066, 0.4840004,  -0.9305912, 0.6281645,
    -0.2873839, -0.0092089, -0.7423917, -0.5064726, 0.2959957,  0.3744118,  -0.2324660, 0.6419766,
    0.0482254,  0.0711853,  -0.0668010, -0.6056250, -0.6424942, 0.5091138,  -0.7920839, -0.3631541,
    0.2925649,  0.8553973,  -0.5368195, -0.8043768, 0.6299060,  -0.7402435, 0.7831608,  -0.4979353,
    -0.7786197, 0.1855255,  -0.7243119, 0.7581270,  0.7850708,  -0.6414960, -0.4423507, -0.4211898,
    0.8494025,  0.3603602,  -0.3777632, 0.3322407,  -0.0483915, -0.8515641, -0.9453503, -0.4536391,
    -0.1080792, 0.5246211,  0.2128397,  -0.0146389, -0.7508293, -0.0058518, 0.5420505,  0.1439000,
    0.1900943,  0.0454271,  0.3117409,  0.1234926,  -0.1166942, 0.2856016,  0.8390452,  0.8877837,
    0.0886838,  -0.7009126, -0.5130350, -0.0999212, 0.3338176,  -0.3013774, 0.3526511,  0.9518843,
    0.5853393,  -0.1422507, -0.9768327, -0.5915277, 0.9691055,  0.4186211,  0.7512146,  0.5220292,
    -0.1700221, 0.5423641,  0.5864487,  -0.7437551, -0.5076052, -0.8304062, 0.4895252,  0.7349310,
    0.7687441,  0.6319372,  0.7462888,  0.2358095};

  std::vector<float> h_trainlab = {
    -0.7061807, -0.9935827, -1.3077246, -0.3378525, -0.6495246, -2.0123182, 0.0340125,  -0.2089733,
    -0.8786033, -1.3019919, -1.9427123, -1.9624611, -1.0215918, -0.7701042, -2.3890236, -0.6768685,
    -1.5100409, -0.7647975, -0.6509883, -0.9327181, -2.2925701, -1.1547282, -0.0646960, -0.2433849,
    -1.3402845, -1.1222004, -1.8060292, -0.5686744, -0.7949885, -0.7014911, -0.4394445, -0.6407220,
    -0.7567281, -0.1424980, -0.4449957, -0.0832827, -1.3135824, -0.7259869, -0.6223005, -1.4591261,
    -1.5859294, -0.7344378, -0.3131946, -0.8229243, -1.1158352, -0.4810999, -0.6265636, -0.9763480,
    -1.3232699, -1.0156538, -0.3958369, -2.3411706, -1.6622960, -0.4680720, -2.0089384, -0.7158608,
    -0.3735971, -1.0591518, -0.3007601, -1.9814152, -1.0727452, -0.7844243, -2.3594606, -0.4388914,
    -0.1194218, -0.4284076, -0.7608060, -0.7356959, -0.7563467, -1.8871661, -2.3971652, -0.4424445,
    -0.7512620, -0.2262175, -0.7759824, -2.5211585, -0.8688839, -0.0325217, -2.0756457, -2.5935947,
    -1.1262706, -0.7814806, -2.6152479, -0.5979422, -1.8219779, -1.2011619, -0.9094200, -1.1892029,
    -0.6205842, -1.7599165, -1.9918835, -0.7041349, -0.7746859, -0.6861359, -0.5224625, -1.2406723,
    -0.1745701, -0.1291239, -2.4182146, -0.5995310, -1.1388247, -0.8812391, -1.1353377, -1.5786207,
    -0.5555833, 0.0002464,  -0.1457169, -1.1594313, -2.1163798, -1.1098294, -1.4213709, -0.4476795,
    -1.5073204, -0.2717116, -0.6787519, -0.8713962, -0.9872876, -0.3698685, 0.0235867,  -1.0940261,
    -0.8272783, -1.9253905, -0.1709152, -0.6209573, -0.5865176, -0.7986188, -2.1974506, -2.6496017,
    -1.9451187, -0.7424771, -1.8817208, -2.2417800, -0.8650095, -0.7006861, -2.0289972, -1.3193644,
    -1.8613344, -1.0139089, -0.7310213, -0.5095533, -0.2320652, -2.3944243, 0.0525441,  -0.5716605,
    -0.0658016, -1.4066644, -0.6430519, -0.5938018, -0.6804599, -0.1180739, -1.7033852, -1.3027941,
    -0.6082652, -2.4703887, -0.9920609, -0.3844494, -0.7468968, 0.0337840,  -0.7998180, -0.0037226,
    -0.5870786, -0.7766853, -0.3147676, -0.7173055, -2.7734269, -0.0547125, -0.4775438, -0.9444610,
    -1.4637991, -1.7066195, -0.0135983, -0.6795068, -1.2210661, -0.1762879, -0.9427360, -0.4120364,
    -0.6077851, -1.7033054, -1.9354388, -0.6399003, -2.1621227, -1.4899510, -0.5816087, 0.0662278,
    -1.7709871, -2.2943379, 0.0671570,  -2.2462875, -0.8166682, -1.3488045, -2.3724372, -0.6542480,
    -1.6837887, 0.1718501,  -0.4232655, -1.9293420, -1.5524519, -0.8903348, -0.8235148, -0.7555137,
    -1.2672423, -0.5341824, -0.0800176, -1.8341924, -2.0388451, -1.6274120, -1.0832978, -0.6836474,
    -0.7428981, -0.6488642, -2.2992384, -0.3173651, -0.6495681, 0.0820371,  -0.2221419, -0.2825119,
    -0.4779604, -0.5677801, -0.5407600, 0.1339569,  -0.8549058, -0.7177885, -0.4706391, -2.0992089,
    -1.7748856, -0.8790807, -0.3359026, -1.0437502, -0.7428065, -0.5449560, 0.2120406,  -0.8962944,
    -2.9057635, -1.8338823, -0.9476171, 0.0537955,  -0.7746540, -0.6021839, -0.9673201, -0.7290961,
    -0.7500160, -2.1319913, -1.6356984, -2.4347284, -0.4906021, -0.1930180, -0.7118280, -0.6601136,
    0.1714188,  -0.4826550};

  std::vector<float> h_test = {
    0.6506153,  -0.2861214, -0.4207479, -0.0879224, 0.6963105,  0.7591472,  -0.9145728, 0.3606104,
    0.5918564,  -0.5548665, -0.4487113, 0.0824032,  0.4425484,  -0.9139633, -0.7823172, 0.0768981,
    0.0922035,  -0.0138858, 0.9646097,  0.2624208,  -0.7190498, -0.6117298, -0.8807327, 0.2868101,
    -0.8899322, 0.9853774,  -0.5898669, 0.6281458,  0.5219784,  -0.5437135, -0.2806136, -0.0927834,
    -0.2291698, 0.0450774,  0.4253027,  0.6545525,  0.7031374,  -0.3601150, 0.0715214,  -0.9844534,
    -0.8571354, -0.8157709, -0.6361769, -0.5510336, 0.4286138,  0.8863587,  -0.7481151, -0.6144726,
    -0.7920206, -0.2917536, -0.6506116, -0.4862449, -0.0866336, -0.7439836, 0.3753550,  0.2632956,
    -0.2270555, 0.1109649,  -0.6320683, 0.0280535,  0.6881603,  0.8163167,  0.1781434,  -0.8063828,
    0.8032009,  -0.6779581, -0.8654890, -0.5322430, 0.3786414,  0.0546245,  -0.5542659, 0.6897840,
    -0.1039676, -0.0343101, 0.4219748,  -0.4535081, 0.7228620,  0.3873561,  0.1427819,  -0.2881901,
    0.5431166,  -0.0090170, -0.8354108, -0.0099369, -0.5904349, 0.2928394,  0.3634137,  -0.7485119,
    -0.5442900, 0.4072478,  -0.4909732, 0.0737537,  -0.0973075, -0.0848911, 0.7041450,  0.3288523,
    -0.5264588, -0.5135713, 0.5130192,  -0.0708379};

  std::vector<float> h_testlab = {
    -1.6506068, -1.6408135, -0.9171102, -2.2897648, -0.2806881, -0.2297245, -0.4421663, -0.7713085,
    -1.6812845, -0.6648566, -0.5840624, -0.8432659, -0.6577426, -1.6213072, -0.2299105, -2.1316719,
    -2.6060586, -1.8153329, 0.1657440,  -0.8794947, -1.3444440, -0.4118046, -0.3390867, -0.9532273,
    0.0358915,  -0.6882091, -0.4517245, -0.3681215, -0.6051433, -1.0756192, -0.6731151, -1.0004896,
    -2.4808031, -1.0080036, -1.7581659, -0.3644765, -0.2742536, -2.1790992, -1.8354263, 0.2105456,
    -0.9973469, -0.2662037, -0.7020552, -0.7884595, -0.6079654, 0.0063403,  -1.2439414, -1.3997503,
    -0.1228729, -0.9907357

  };

  std::vector<float> h_trainwts;
  std::vector<float> h_testwts;

  rmm::device_uvector<float> d_train;
  rmm::device_uvector<float> d_trainlab;
  rmm::device_uvector<float> d_test;
  rmm::device_uvector<float> d_testlab;
  rmm::device_uvector<float> d_trainwts;
  rmm::device_uvector<float> d_testwts;
};

TEST_F(GeneticEvolutionTest, SymReg)
{
  MLCommon::CompareApprox<float> compApprox(tolerance);
  program_t final_progs;
  final_progs = (program_t)rmm::mr::get_current_device_resource()->allocate(
    hyper_params.population_size * sizeof(program), stream);
  std::vector<std::vector<program>> history;
  history.reserve(hyper_params.generations);

  cudaEvent_t start, stop;
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));

  cudaEventRecord(start, stream);

  symFit(handle,
         d_train.data(),
         d_trainlab.data(),
         d_trainwts.data(),
         n_tr_rows,
         n_cols,
         hyper_params,
         final_progs,
         history);

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float training_time;
  cudaEventElapsedTime(&training_time, start, stop);

  int n_gen = history.size();
  std::cout << "Finished training for " << n_gen << " generations." << std::endl;

  // Find index of best program
  int best_idx      = 0;
  float opt_fitness = history[n_gen - 1][0].raw_fitness_;

  // For all 3 loss functions - min is better
  for (int i = 1; i < hyper_params.population_size; ++i) {
    if (history[n_gen - 1][i].raw_fitness_ < opt_fitness) {
      best_idx    = i;
      opt_fitness = history[n_gen - 1][i].raw_fitness_;
    }
  }

  std::string eqn = stringify(history[n_gen - 1][best_idx]);
  CUML_LOG_DEBUG("Best Index = %d", best_idx);
  std::cout << "Raw fitness score on train set is " << history[n_gen - 1][best_idx].raw_fitness_
            << std::endl;
  std::cout << "Best AST equation is : " << eqn << std::endl;

  // Predict values for test dataset
  rmm::device_uvector<float> d_predlabels(n_tst_rows, stream);

  cudaEventRecord(start, stream);

  cuml::genetic::symRegPredict(
    handle, d_test.data(), n_tst_rows, final_progs + best_idx, d_predlabels.data());

  std::vector<float> h_predlabels(n_tst_rows, 0.0f);
  RAFT_CUDA_TRY(cudaMemcpy(
    h_predlabels.data(), d_predlabels.data(), n_tst_rows * sizeof(float), cudaMemcpyDeviceToHost));

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float inference_time;
  cudaEventElapsedTime(&inference_time, start, stop);

  // deallocate the nodes allocated for the last generation inside SymFit
  for (auto i = 0; i < hyper_params.population_size; ++i) {
    program tmp = program();
    raft::copy(&tmp, final_progs + i, 1, stream);
    rmm::mr::get_current_device_resource()->deallocate(tmp.nodes, tmp.len * sizeof(node), stream);
    tmp.nodes = nullptr;
  }
  // deallocate the final programs from device memory
  rmm::mr::get_current_device_resource()->deallocate(
    final_progs, hyper_params.population_size * sizeof(program), stream);

  ASSERT_TRUE(compApprox(history[n_gen - 1][best_idx].raw_fitness_, 0.0036f));
  std::cout << "Some Predicted test values:" << std::endl;
  std::copy(
    h_predlabels.begin(), h_predlabels.begin() + 10, std::ostream_iterator<float>(std::cout, ";"));
  std::cout << std::endl;

  std::cout << "Some Actual test values:" << std::endl;
  std::copy(
    h_testlab.begin(), h_testlab.begin() + 10, std::ostream_iterator<float>(std::cout, ";"));
  std::cout << std::endl;

  std::cout << "Training time = " << training_time << " ms" << std::endl;
  std::cout << "Inference time = " << inference_time << " ms" << std::endl;
}

}  // namespace genetic
}  // namespace cuml
