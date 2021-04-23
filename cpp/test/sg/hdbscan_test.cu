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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <vector>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/linalg/distance_type.h>
#include <raft/linalg/transpose.h>
#include <raft/mr/device/allocator.hpp>
#include <raft/sparse/coo.cuh>
#include <rmm/device_uvector.hpp>

#include "../prims/test_utils.h"

namespace ML {

using namespace std;

template <typename T, typename IdxT>
struct LinkageInputs {
  IdxT n_row;
  IdxT n_col;
  int k, min_pts, min_cluster_size;

  std::vector<T> data;

  std::vector<IdxT> expected_labels;
};

/**
* @brief kernel to calculate the values of a and b
* @param firstClusterArray: the array of classes of type T
* @param secondClusterArray: the array of classes of type T
* @param size: the size of the data points
* @param a: number of pairs of points that both the clusters have classified the same
* @param b: number of pairs of points that both the clusters have classified differently
*/
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void computeTheNumerator(const T* firstClusterArray,
                                    const T* secondClusterArray, int size,
                                    int* a, int* b) {
  //calculating the indices of pairs of datapoints compared by the current thread
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  //thread-local variables to count a and b
  int myA = 0, myB = 0;

  if (i < size && j < size && j < i) {
    //checking if the pair have been classified the same by both the clusters
    if (firstClusterArray[i] == firstClusterArray[j] &&
        secondClusterArray[i] == secondClusterArray[j]) {
      ++myA;
    }

    //checking if the pair have been classified differently by both the clusters
    else if (firstClusterArray[i] != firstClusterArray[j] &&
             secondClusterArray[i] != secondClusterArray[j]) {
      ++myB;
    }
  }

  //specialize blockReduce for a 2D block of 1024 threads of type int
  typedef cub::BlockReduce<int, BLOCK_DIM_X,
                           cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
    BlockReduce;

  //Allocate shared memory for blockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  //summing up thread-local counts specific to a block
  myA = BlockReduce(temp_storage).Sum(myA);
  __syncthreads();
  myB = BlockReduce(temp_storage).Sum(myB);
  __syncthreads();

  //executed once per block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    raft::myAtomicAdd<unsigned long long int>((unsigned long long int*)a, myA);
    raft::myAtomicAdd<unsigned long long int>((unsigned long long int*)b, myB);
  }
}

/**
* @brief Function to calculate RandIndex
* <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
* @param firstClusterArray: the array of classes of type T
* @param secondClusterArray: the array of classes of type T
* @param size: the size of the data points of type int
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename T>
double compute_rand_index(
  T* firstClusterArray, T* secondClusterArray, int size,
  std::shared_ptr<raft::mr::device::allocator> allocator, cudaStream_t stream) {
  //rand index for size less than 2 is not defined
  ASSERT(size >= 2, "Rand Index for size less than 2 not defined!");

  //allocating and initializing memory for a and b in the GPU
  raft::mr::device::buffer<int> arr_buf(allocator, stream, 2);
  CUDA_CHECK(cudaMemsetAsync(arr_buf.data(), 0, 2 * sizeof(int), stream));

  //kernel configuration
  static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(raft::ceildiv<int>(size, numThreadsPerBlock.x),
                 raft::ceildiv<int>(size, numThreadsPerBlock.y));

  //calling the kernel
  computeTheNumerator<T, BLOCK_DIM_X, BLOCK_DIM_Y>
    <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      firstClusterArray, secondClusterArray, size, arr_buf.data(),
      arr_buf.data() + 1);

  //synchronizing and updating the calculated values of a and b from device to host
  int ab_host[2] = {0};
  raft::update_host(ab_host, arr_buf.data(), 2, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //error handling
  CUDA_CHECK(cudaGetLastError());

  //denominator
  int nChooseTwo = size * (size - 1) / 2;

  //calculating the rand_index
  return (double)(((double)(ab_host[0] + ab_host[1])) / (double)nChooseTwo);
}

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os,
                           const LinkageInputs<T, IdxT>& dims) {
  return os;
}

template <typename T, typename IdxT>
class HDBSCANTest : public ::testing::TestWithParam<LinkageInputs<T, IdxT>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<LinkageInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<T> data(params.n_row * params.n_col,
                                handle.get_stream());

    // Allocate result labels and expected labels on device
    raft::allocate(labels, params.n_row);
    raft::allocate(labels_ref, params.n_row);

    raft::copy(data.data(), params.data.data(), data.size(),
               handle.get_stream());
    raft::copy(labels_ref, params.expected_labels.data(), params.n_row,
               handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2,
                                           handle.get_stream());

    auto* output = new hdbscan_output<IdxT, T>();
    hdbscan(handle, data.data(), params.n_row, params.n_col,
            raft::distance::DistanceType::L2SqrtExpanded, params.k,
            params.min_pts, params.min_cluster_size, output);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    score =
      compute_rand_index(labels, labels_ref, params.n_row,
                         handle.get_device_allocator(), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(labels_ref));
  }

 protected:
  LinkageInputs<T, IdxT> params;
  IdxT *labels, *labels_ref;
  int k;

  double score;
};

const std::vector<LinkageInputs<float, int>> hdbscan_inputsf2 = {
  // Test n_clusters == n_points
  {10,
   5,
   5,
   2,
   2,
   {0.21390334, 0.50261639, 0.91036676, 0.59166485, 0.71162682, 0.10248392,
    0.77782677, 0.43772379, 0.4035871,  0.3282796,  0.47544681, 0.59862974,
    0.12319357, 0.06239463, 0.28200272, 0.1345717,  0.50498218, 0.5113505,
    0.16233086, 0.62165332, 0.42281548, 0.933117,   0.41386077, 0.23264562,
    0.73325968, 0.37537541, 0.70719873, 0.14522645, 0.73279625, 0.9126674,
    0.84854131, 0.28890216, 0.85267903, 0.74703138, 0.83842071, 0.34942792,
    0.27864171, 0.70911132, 0.21338564, 0.32035554, 0.73788331, 0.46926692,
    0.57570162, 0.42559178, 0.87120209, 0.22734951, 0.01847905, 0.75549396,
    0.76166195, 0.66613745},
   {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
  //  // Test outlier points
  {9,
   2,
   3,
   3,
   3,
   {-1, -50, 3, 4, 5000, 10000, 1, 3, 4, 5, 0.000005, 0.00002, 2000000, 500000,
    10, 50, 30, 5},
   {6, 0, 5, 0, 0, 4, 3, 2, 1}},

  // Test n_clusters == (n_points / 2)
  {10,
   5,
   4,
   3,
   4,
   {0.21390334, 0.50261639, 0.91036676, 0.59166485, 0.71162682, 0.10248392,
    0.77782677, 0.43772379, 0.4035871,  0.3282796,  0.47544681, 0.59862974,
    0.12319357, 0.06239463, 0.28200272, 0.1345717,  0.50498218, 0.5113505,
    0.16233086, 0.62165332, 0.42281548, 0.933117,   0.41386077, 0.23264562,
    0.73325968, 0.37537541, 0.70719873, 0.14522645, 0.73279625, 0.9126674,
    0.84854131, 0.28890216, 0.85267903, 0.74703138, 0.83842071, 0.34942792,
    0.27864171, 0.70911132, 0.21338564, 0.32035554, 0.73788331, 0.46926692,
    0.57570162, 0.42559178, 0.87120209, 0.22734951, 0.01847905, 0.75549396,
    0.76166195, 0.66613745},
   {1, 0, 4, 0, 0, 3, 2, 0, 2, 1}},

  // Test n_points == 100
  {100,
   10,
   10,
   10,
   25,
   {6.26168372e-01, 9.30437651e-01, 6.02450208e-01,
    2.73025296e-01, 9.53050619e-01, 3.32164396e-01,
    6.88942598e-01, 5.79163537e-01, 6.70341547e-01,
    2.70140602e-02, 9.30429671e-01, 7.17721157e-01,
    9.89948537e-01, 7.75253347e-01, 1.34491522e-02,
    2.48522428e-02, 3.51413378e-01, 7.64405834e-01,
    7.86373507e-01, 7.18748577e-01, 8.66998621e-01,
    6.80316582e-01, 2.51288712e-01, 4.91078420e-01,
    3.76246281e-01, 4.86828710e-01, 5.67464772e-01,
    5.30734742e-01, 8.99478296e-01, 7.66699088e-01,
    9.49339111e-01, 3.55248484e-01, 9.06046929e-01,
    4.48407772e-01, 6.96395305e-01, 2.44277335e-01,
    7.74840000e-01, 5.21046603e-01, 4.66423971e-02,
    5.12019638e-02, 8.95019614e-01, 5.28956953e-01,
    4.31536306e-01, 5.83857744e-01, 4.41787364e-01,
    4.68656523e-01, 5.73971433e-01, 6.79989654e-01,
    3.19650588e-01, 6.12579596e-01, 6.49126442e-02,
    8.39131142e-01, 2.85252117e-01, 5.84848929e-01,
    9.46507115e-01, 8.58440748e-01, 3.61528940e-01,
    2.44215959e-01, 3.80101125e-01, 4.57128957e-02,
    8.82216988e-01, 8.31498633e-01, 7.23474381e-01,
    7.75788607e-01, 1.40864146e-01, 6.62092382e-01,
    5.13985168e-01, 3.00686418e-01, 8.70109949e-01,
    2.43187753e-01, 2.89391938e-01, 2.84214238e-01,
    8.70985521e-01, 8.77491176e-01, 6.72537226e-01,
    3.30929686e-01, 1.85934324e-01, 9.16222614e-01,
    6.18239142e-01, 2.64768597e-01, 5.76145451e-01,
    8.62961369e-01, 6.84757925e-01, 7.60549082e-01,
    1.27645356e-01, 4.51004673e-01, 3.92292980e-01,
    4.63170803e-01, 4.35449330e-02, 2.17583404e-01,
    5.71832605e-02, 2.06763039e-01, 3.70116249e-01,
    2.09750028e-01, 6.17283019e-01, 8.62549231e-01,
    9.84156240e-02, 2.66249156e-01, 3.87635103e-01,
    2.85591012e-02, 4.24826068e-01, 4.45795088e-01,
    6.86227676e-01, 1.08848960e-01, 5.96731841e-02,
    3.71770228e-01, 1.91548833e-01, 6.95136078e-01,
    9.00700636e-01, 8.76363105e-01, 2.67334632e-01,
    1.80619709e-01, 7.94060419e-01, 1.42854171e-02,
    1.09372387e-01, 8.74028108e-01, 6.46403232e-01,
    4.86588834e-01, 5.93446175e-02, 6.11886291e-01,
    8.83865057e-01, 3.15879821e-01, 2.27043992e-01,
    9.76764951e-01, 6.15620336e-01, 9.76199360e-01,
    2.40548962e-01, 3.21795663e-01, 8.75087904e-02,
    8.11234663e-01, 6.96070480e-01, 8.12062321e-01,
    1.21958818e-01, 3.44348628e-02, 8.72630414e-01,
    3.06162776e-01, 1.76043529e-02, 9.45894971e-01,
    5.33896401e-01, 6.21642973e-01, 4.93062535e-01,
    4.48984262e-01, 2.24560379e-01, 4.24052195e-02,
    4.43447610e-01, 8.95646149e-01, 6.05220676e-01,
    1.81840491e-01, 9.70831206e-01, 2.12563586e-02,
    6.92582693e-01, 7.55946922e-01, 7.95086143e-01,
    6.05328941e-01, 3.99350764e-01, 4.32846636e-01,
    9.81114529e-01, 4.98266428e-01, 6.37127930e-03,
    1.59085889e-01, 6.34682067e-05, 5.59429440e-01,
    7.38827633e-01, 8.93214770e-01, 2.16494306e-01,
    9.35430573e-02, 4.75665868e-02, 7.80503518e-01,
    7.86240041e-01, 7.06854594e-01, 2.13725879e-02,
    7.68246091e-01, 4.50234808e-01, 5.21231104e-01,
    5.01989826e-03, 4.22081572e-02, 1.65337732e-01,
    8.54134740e-01, 4.99430262e-01, 8.94525601e-01,
    1.14028379e-01, 3.69739861e-01, 1.32955599e-01,
    2.65563824e-01, 2.52811151e-01, 1.44792843e-01,
    6.88449594e-01, 4.44921417e-01, 8.23296587e-01,
    1.93266317e-01, 1.19033309e-01, 1.36368966e-01,
    3.42600285e-01, 5.64505195e-01, 5.57594559e-01,
    7.44257892e-01, 8.38231569e-02, 4.11548847e-01,
    3.21010077e-01, 8.55081359e-01, 4.30105779e-01,
    1.16229135e-01, 9.87731964e-02, 3.14712335e-01,
    4.50880592e-01, 2.72289598e-01, 6.31615256e-01,
    8.97432958e-01, 4.44764250e-01, 8.03776440e-01,
    2.68767748e-02, 2.43374608e-01, 4.02141103e-01,
    4.98881209e-01, 5.33173003e-01, 8.82890436e-01,
    7.16149148e-01, 4.19664401e-01, 2.29335357e-01,
    2.88637806e-01, 3.44696803e-01, 6.78171906e-01,
    5.69849716e-01, 5.86454477e-01, 3.54474989e-01,
    9.03876540e-01, 6.45980000e-01, 6.34887593e-01,
    7.88039746e-02, 2.04814126e-01, 7.82251754e-01,
    2.43147074e-01, 7.50951808e-01, 1.72799092e-02,
    2.95349590e-01, 6.57991826e-01, 8.81214312e-01,
    5.73970708e-01, 2.77610881e-01, 1.82155097e-01,
    7.69797417e-02, 6.44792402e-01, 9.46950998e-01,
    7.73064845e-01, 6.04733624e-01, 5.80094567e-01,
    1.67498426e-01, 2.66514296e-01, 6.50140368e-01,
    1.91170299e-01, 2.08752199e-01, 3.01664091e-01,
    9.85033484e-01, 2.92909152e-01, 8.65816607e-01,
    1.85222119e-01, 2.28814559e-01, 1.34286382e-02,
    2.89234322e-01, 8.18668708e-01, 4.71706924e-01,
    9.23199803e-01, 2.80879188e-01, 1.47319284e-01,
    4.13915748e-01, 9.31274932e-02, 6.66322195e-01,
    9.66953974e-01, 3.19405786e-01, 6.69486551e-01,
    5.03096313e-02, 6.95225201e-01, 5.78469859e-01,
    6.29481655e-01, 1.39252534e-01, 1.22564968e-01,
    6.80663678e-01, 6.34607157e-01, 6.42765834e-01,
    1.57127410e-02, 2.92132086e-01, 5.24423878e-01,
    4.68676824e-01, 2.86003928e-01, 7.18608322e-01,
    8.95617933e-01, 5.48844309e-01, 1.74517278e-01,
    5.24379196e-01, 2.13526524e-01, 5.88375435e-01,
    9.88560185e-01, 4.17435771e-01, 6.14438688e-01,
    9.53760881e-01, 5.27151288e-01, 7.03017278e-01,
    3.44448559e-01, 4.47059676e-01, 2.83414901e-01,
    1.98979011e-01, 4.24917361e-01, 5.73172761e-01,
    2.32398853e-02, 1.65887230e-01, 4.05552785e-01,
    9.29665524e-01, 2.26135696e-01, 9.20563384e-01,
    7.65259963e-01, 4.54820075e-01, 8.97710267e-01,
    3.78559302e-03, 9.15219382e-01, 3.55705698e-01,
    6.94905124e-01, 8.58540202e-01, 3.89790666e-01,
    2.49478206e-01, 7.93679304e-01, 4.75830027e-01,
    4.40425353e-01, 3.70579459e-01, 1.40578049e-01,
    1.70386675e-01, 7.04056121e-01, 4.85963102e-01,
    9.68450060e-01, 6.77178001e-01, 2.65934654e-01,
    2.58915007e-01, 6.70052890e-01, 2.61945109e-01,
    8.46207759e-01, 1.01928951e-01, 2.85611334e-01,
    2.45776933e-01, 2.66658783e-01, 3.71724077e-01,
    4.34319025e-01, 4.24407347e-01, 7.15417683e-01,
    8.07997684e-01, 1.64296275e-01, 6.01638065e-01,
    8.60606804e-02, 2.68719187e-01, 5.11764101e-01,
    9.75844338e-01, 7.81226782e-01, 2.20925515e-01,
    7.18135040e-01, 9.82395577e-01, 8.39160243e-01,
    9.08058083e-01, 6.88010677e-01, 8.14271847e-01,
    5.12460821e-01, 1.17311345e-01, 5.96075228e-01,
    9.17455497e-01, 2.12052706e-01, 7.04074603e-01,
    8.72872565e-02, 8.76047818e-01, 6.96235046e-01,
    8.54801557e-01, 2.49729159e-01, 9.76594604e-01,
    2.87386363e-01, 2.36461559e-02, 9.94075254e-01,
    4.25193986e-01, 7.61869994e-01, 5.13334255e-01,
    6.44711165e-02, 8.92156689e-01, 3.55235167e-01,
    1.08154647e-01, 8.78446825e-01, 2.43833016e-01,
    9.23071293e-01, 2.72724115e-01, 9.46631338e-01,
    3.74510294e-01, 4.08451278e-02, 9.78392777e-01,
    3.65079221e-01, 6.37199516e-01, 5.51144906e-01,
    5.25978080e-01, 1.42803678e-01, 4.05451674e-01,
    7.79788219e-01, 6.26009784e-01, 3.35249497e-01,
    1.43159543e-02, 1.80363779e-01, 5.05096904e-01,
    2.82619947e-01, 5.83561392e-01, 3.10951324e-01,
    8.73223968e-01, 4.38545619e-01, 4.81348800e-01,
    6.68497085e-01, 3.79345401e-01, 9.58832501e-01,
    1.89869550e-01, 2.34083070e-01, 2.94066207e-01,
    5.74892667e-02, 6.92106828e-02, 9.61127686e-02,
    6.72650672e-02, 8.47345378e-01, 2.80916761e-01,
    7.32177357e-03, 9.80785961e-01, 5.73192225e-02,
    8.48781331e-01, 8.83225408e-01, 7.34398275e-01,
    7.70381941e-01, 6.20778343e-01, 8.96822048e-01,
    5.40732486e-01, 3.69704071e-01, 5.77305837e-01,
    2.08221827e-01, 7.34275341e-01, 1.06110900e-01,
    3.49496706e-01, 8.34948910e-01, 1.56403291e-02,
    6.78576376e-01, 8.96141268e-01, 5.94835119e-01,
    1.43943153e-01, 3.49618530e-01, 2.10440392e-01,
    3.46585620e-01, 1.05153093e-01, 3.45446174e-01,
    2.72177079e-01, 7.07946300e-01, 4.33717726e-02,
    3.31232203e-01, 3.91874320e-01, 4.76338141e-01,
    6.22777789e-01, 2.95989228e-02, 4.32855769e-01,
    7.61049310e-01, 3.63279149e-01, 9.47210350e-01,
    6.43721247e-01, 6.58025802e-01, 1.05247633e-02,
    5.29974442e-01, 7.30675767e-01, 4.30041079e-01,
    6.62634841e-01, 8.25936616e-01, 9.91253704e-01,
    6.79399281e-01, 5.44177006e-01, 7.52876048e-01,
    3.32139049e-01, 7.98732398e-01, 7.38865223e-01,
    9.16055132e-01, 6.11736493e-01, 9.63672879e-01,
    1.83778839e-01, 7.27558919e-02, 5.91602822e-01,
    3.25235484e-01, 2.34741217e-01, 9.52346277e-01,
    9.18556407e-01, 9.35373324e-01, 6.89209070e-01,
    2.56049054e-01, 6.17975395e-01, 7.82285691e-01,
    9.84983432e-01, 6.62322741e-01, 2.04144457e-01,
    3.98446577e-01, 1.38918297e-01, 3.05919921e-01,
    3.14043787e-01, 5.91072666e-01, 7.44703771e-01,
    8.92272567e-01, 9.78017873e-01, 9.01203161e-01,
    1.41526372e-01, 4.14878484e-01, 6.80683651e-01,
    5.01733152e-02, 8.14635389e-01, 2.27926375e-01,
    9.03269815e-01, 8.68443745e-01, 9.86939190e-01,
    7.40779486e-01, 2.61005311e-01, 3.19276232e-01,
    9.69509248e-01, 1.11908818e-01, 4.49198556e-01,
    1.27056715e-01, 3.84064823e-01, 5.14591811e-01,
    2.10747488e-01, 9.53884090e-01, 8.43167950e-01,
    4.51187972e-01, 3.75331782e-01, 6.23566461e-01,
    3.55290379e-01, 2.95705968e-01, 1.69622690e-01,
    1.42981830e-01, 2.72180991e-01, 9.46468040e-01,
    3.70932500e-01, 9.94292830e-01, 4.62587505e-01,
    7.14817405e-01, 2.45370540e-02, 3.00906377e-01,
    5.75768304e-01, 9.71448393e-01, 6.95574827e-02,
    3.93693854e-01, 5.29306116e-01, 5.04694554e-01,
    6.73797120e-02, 6.76596969e-01, 5.50948898e-01,
    3.24909641e-01, 7.70337719e-01, 6.51842631e-03,
    3.03264879e-01, 7.61037886e-03, 2.72289601e-01,
    1.50502041e-01, 6.71103888e-02, 7.41503703e-01,
    1.92088941e-01, 2.19043977e-01, 9.09320161e-01,
    2.37993569e-01, 6.18107973e-02, 8.31447852e-01,
    2.23355609e-01, 1.84789435e-01, 4.16104518e-01,
    4.21573859e-01, 8.72446305e-02, 2.97294197e-01,
    4.50328256e-01, 8.72199917e-01, 2.51279916e-01,
    4.86219272e-01, 7.57071329e-01, 4.85655942e-01,
    1.06187277e-01, 4.92341327e-01, 1.46017513e-01,
    5.25421017e-01, 4.22637906e-01, 2.24685018e-01,
    8.72648431e-01, 5.54051490e-01, 1.80745062e-01,
    2.12756336e-01, 5.20883169e-01, 7.60363654e-01,
    8.30254678e-01, 5.00003328e-01, 4.69017439e-01,
    6.38105527e-01, 3.50638261e-02, 5.22217353e-02,
    9.06516882e-02, 8.52975842e-01, 1.19985883e-01,
    3.74926753e-01, 6.50302066e-01, 1.98875727e-01,
    6.28362507e-02, 4.32693501e-01, 3.10500685e-01,
    6.20732833e-01, 4.58503272e-01, 3.20790034e-01,
    7.91284868e-01, 7.93054570e-01, 2.93406765e-01,
    8.95399023e-01, 1.06441034e-01, 7.53085241e-02,
    8.67523104e-01, 1.47963482e-01, 1.25584706e-01,
    3.81545040e-02, 6.34338619e-01, 1.76368938e-02,
    5.75553531e-02, 5.31607516e-01, 2.63869588e-01,
    9.41945823e-01, 9.24028838e-02, 5.21496463e-01,
    7.74866558e-01, 5.65210610e-01, 7.28015327e-02,
    6.51963790e-01, 8.94727453e-01, 4.49571590e-01,
    1.29932405e-01, 8.64026259e-01, 9.92599934e-01,
    7.43721560e-01, 8.87300215e-01, 1.06369925e-01,
    8.11335531e-01, 7.87734900e-01, 9.87344678e-01,
    5.32502820e-01, 4.42612382e-01, 9.64041183e-01,
    1.66085871e-01, 1.12937664e-01, 5.24423470e-01,
    6.54689333e-01, 4.59119726e-01, 5.22774091e-01,
    3.08722276e-02, 6.26979315e-01, 4.49754105e-01,
    8.07495757e-01, 2.34199499e-01, 1.67765675e-01,
    9.22168418e-01, 3.73210378e-01, 8.04432575e-01,
    5.61890354e-01, 4.47025593e-01, 6.43155678e-01,
    2.40407640e-01, 5.91631279e-01, 1.59369206e-01,
    7.75799090e-01, 8.32067212e-01, 5.59791576e-02,
    6.39105224e-01, 4.85274738e-01, 2.12630838e-01,
    2.81431312e-02, 7.16205363e-01, 6.83885011e-01,
    5.23869697e-01, 9.99418314e-01, 8.35331599e-01,
    4.69877463e-02, 6.74712562e-01, 7.99273684e-01,
    2.77001890e-02, 5.75809742e-01, 2.78513031e-01,
    8.36209905e-01, 7.25472379e-01, 4.87173943e-01,
    7.88311357e-01, 9.64676177e-01, 1.75752651e-01,
    4.98112580e-01, 8.08850418e-02, 6.40981131e-01,
    4.06647450e-01, 8.46539387e-01, 2.12620694e-01,
    9.11012851e-01, 8.25041445e-01, 8.90065575e-01,
    9.63626055e-01, 5.96689242e-01, 1.63372670e-01,
    4.51640148e-01, 3.43026542e-01, 5.80658851e-01,
    2.82327625e-01, 4.75535418e-01, 6.27760926e-01,
    8.46314115e-01, 9.61961932e-01, 3.19806094e-01,
    5.05508062e-01, 5.28102944e-01, 6.13045057e-01,
    7.44714938e-01, 1.50586073e-01, 7.91878033e-01,
    4.89839179e-01, 3.10496849e-01, 8.82309038e-01,
    2.86922314e-01, 4.84687559e-01, 5.20838630e-01,
    4.62955493e-01, 2.38185305e-01, 5.47259907e-02,
    7.10916137e-01, 7.31887202e-01, 6.25602317e-01,
    8.77741168e-01, 4.19881322e-01, 4.81222328e-01,
    1.28224501e-01, 2.46034010e-01, 3.34971854e-01,
    7.37216484e-01, 5.62134821e-02, 7.14089724e-01,
    9.85549393e-01, 4.66295827e-01, 3.08722434e-03,
    4.70237690e-01, 2.66524167e-01, 7.93875484e-01,
    4.54795911e-02, 8.09702944e-01, 1.47709735e-02,
    1.70082405e-01, 6.35905179e-01, 3.75379109e-01,
    4.30315011e-01, 3.15788760e-01, 5.58065230e-01,
    2.24643800e-01, 2.42142981e-01, 6.57283636e-01,
    3.34921891e-01, 1.26588975e-01, 7.68064155e-01,
    9.43856291e-01, 4.47518596e-01, 5.44453573e-01,
    9.95764932e-01, 7.16444391e-01, 8.51019765e-01,
    1.01179183e-01, 4.45473958e-01, 4.60327322e-01,
    4.96895844e-02, 4.72907738e-01, 5.58987444e-01,
    3.41027487e-01, 1.56175026e-01, 7.58283148e-01,
    6.83600909e-01, 2.14623396e-01, 3.27348880e-01,
    3.92517893e-01, 6.70418431e-01, 5.16440832e-01,
    8.63140348e-01, 5.73277464e-01, 3.46608058e-01,
    7.39396341e-01, 7.20852434e-01, 2.35653246e-02,
    3.89935659e-01, 7.53783745e-01, 6.34563528e-01,
    8.79339335e-01, 7.41599159e-02, 5.62433904e-01,
    6.15553852e-01, 4.56956324e-01, 5.20047447e-01,
    5.26845015e-02, 5.58471266e-01, 1.63632233e-01,
    5.38936665e-02, 6.49593683e-01, 2.56838748e-01,
    8.99035326e-01, 7.20847756e-01, 5.68954684e-01,
    7.43684755e-01, 5.70924238e-01, 3.82318724e-01,
    4.89328290e-01, 5.62208561e-01, 4.97540804e-02,
    4.18011085e-01, 6.88041565e-01, 2.16234653e-01,
    7.89548214e-01, 8.46136387e-01, 8.46816189e-01,
    1.73842353e-01, 6.11627842e-02, 8.44440559e-01,
    4.50646654e-01, 3.74785037e-01, 4.87196697e-01,
    4.56276448e-01, 9.13284391e-01, 4.15715464e-01,
    7.13597697e-01, 1.23641270e-02, 5.10031271e-01,
    4.74601930e-02, 2.55731159e-01, 3.22090006e-01,
    1.91165703e-01, 4.51170940e-01, 7.50843157e-01,
    4.42420576e-01, 4.25380660e-01, 4.50667257e-01,
    6.55689206e-01, 9.68257670e-02, 1.96528793e-01,
    8.97343028e-01, 4.99940904e-01, 6.65504083e-01,
    9.41828079e-01, 4.54397338e-01, 5.61893331e-01,
    5.09839880e-01, 4.53117514e-01, 8.96804127e-02,
    1.74888861e-01, 6.65641378e-01, 2.81668336e-01,
    1.89532742e-01, 5.61668382e-01, 8.68330157e-02,
    8.25092797e-01, 5.18106324e-01, 1.71904024e-01,
    3.68385523e-01, 1.62005436e-01, 7.48507399e-01,
    9.30274827e-01, 2.38198517e-01, 9.52222901e-01,
    5.23587800e-01, 6.94384557e-01, 1.09338652e-01,
    4.83356794e-01, 2.73050402e-01, 3.68027050e-01,
    5.92366466e-01, 1.83192289e-01, 8.60376029e-01,
    7.13926203e-01, 8.16750052e-01, 1.57890291e-01,
    6.25691951e-01, 5.24831646e-01, 1.73873797e-01,
    1.02429784e-01, 9.17488471e-01, 4.03584434e-01,
    9.31170884e-01, 2.79386137e-01, 8.77745206e-01,
    2.45200576e-01, 1.28896951e-01, 3.15713052e-01,
    5.27874291e-01, 2.16444335e-01, 7.03883817e-01,
    7.74738919e-02, 8.42422142e-01, 3.75598924e-01,
    3.51002411e-01, 6.22752776e-01, 4.82407943e-01,
    7.43107867e-01, 9.46182666e-01, 9.44344819e-01,
    3.28124763e-01, 1.06147431e-01, 1.65102684e-01,
    3.84060507e-01, 2.91057722e-01, 7.68173662e-02,
    1.03543651e-01, 6.76698940e-01, 1.43141994e-01,
    7.21342202e-01, 6.69471294e-03, 9.07298311e-01,
    5.57080171e-01, 8.10954489e-01, 4.11120526e-01,
    2.06407453e-01, 2.59590556e-01, 7.58512718e-01,
    5.79873897e-01, 2.92875650e-01, 2.83686529e-01,
    2.42829343e-01, 9.19323719e-01, 3.46832864e-01,
    3.58238858e-01, 7.42827585e-01, 2.05760059e-01,
    9.58438860e-01, 5.66326411e-01, 6.60292846e-01,
    5.61095078e-02, 6.79465531e-01, 7.05118513e-01,
    4.44713264e-01, 2.09732933e-01, 5.22732436e-01,
    1.74396512e-01, 5.29356748e-01, 4.38475687e-01,
    4.94036404e-01, 4.09785794e-01, 6.40025507e-01,
    5.79371821e-01, 1.57726118e-01, 6.04572263e-01,
    5.41072639e-01, 5.18847173e-01, 1.97093284e-01,
    8.91767002e-01, 4.29050835e-01, 8.25490570e-01,
    3.87699807e-01, 4.50705808e-01, 2.49371643e-01,
    3.36074898e-01, 9.29925118e-01, 6.65393649e-01,
    9.07275994e-01, 3.73075859e-01, 4.14044139e-03,
    2.37463702e-01, 2.25893784e-01, 2.46900245e-01,
    4.50350196e-01, 3.48618117e-01, 5.07193932e-01,
    5.23435142e-01, 8.13611417e-01, 8.92715622e-01,
    1.02623450e-01, 3.06088345e-01, 7.80461650e-01,
    2.21453645e-01, 2.01419652e-01, 2.84254457e-01,
    3.68286735e-01, 7.39358243e-01, 8.97879394e-01,
    9.81599566e-01, 7.56526442e-01, 7.37645545e-01,
    4.23976657e-02, 8.25922012e-01, 2.60956996e-01,
    2.90702065e-01, 8.98388344e-01, 3.03733299e-01,
    8.49071471e-01, 3.45835425e-01, 7.65458276e-01,
    5.68094872e-01, 8.93770930e-01, 9.93161641e-01,
    5.63368667e-02, 4.26548945e-01, 5.46745780e-01,
    5.75674571e-01, 7.94599487e-01, 7.18935553e-02,
    4.46492976e-01, 6.40240123e-01, 2.73246969e-01,
    2.00465968e-01, 1.30718835e-01, 1.92492005e-01,
    1.96617189e-01, 6.61271644e-01, 8.12687657e-01,
    8.66342445e-01

   },
   {0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
    4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}};

typedef HDBSCANTest<float, int> HDBSCANTestF_Int;
TEST_P(HDBSCANTestF_Int, Result) {
  //  EXPECT_TRUE(score == 1.0);
}

INSTANTIATE_TEST_CASE_P(HDBSCANTest, HDBSCANTestF_Int,
                        ::testing::ValuesIn(hdbscan_inputsf2));
}  // end namespace ML