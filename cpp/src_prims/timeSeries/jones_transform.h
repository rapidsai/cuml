

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
/**
* @file ar_param_transform.h
* @brief TODO brief
*/

#include <math.h>
#include <cub/cub.cuh>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/unary_op.h"


namespace MLCommon {

namespace TimeSeries {

//just a helper function to display stuff
template <typename DataT>
void display_helper(DataT *arr, int row, int col, cudaStream_t stream){

  DataT *h_arr = (DataT*) malloc(row*col*sizeof(DataT*));

  updateHost(h_arr, arr, row*col, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  for(int i = 0; i<row; ++i){
    for(int j=0; j<col; ++j){
      printf("%f ",h_arr[i*col+j]);
    }
    printf("\n");
  }
}



/**
* @brief Lambda to map to the partial autocorrelation
*
* @tparam Type: Data type of the input 
* @param in: the input to the functional mapping
* @return : the Partial autocorrelation (ie, tanh(in/2))
*/
template <typename Type>
struct PAC {
  HDI Type operator()(Type in) { return ((1- myExp(-1*in))/(1+myExp(-1*in))); }
};


/**
* @brief Lambda to map to the arctanh
*
* @tparam Type: Data type of the input 
* @param in: the input to the functional mapping
* @return : arctanh() of the input
*/
template <typename Type>
struct arctanh {
  HDI Type operator()(Type in) { return (log(1+in) - log(1-in))/2; }
};

/**
 * @brief kernel to perform jones inverse transformation on the autoregressive params
 * @tparam DataT: type of the params
 * @tparam P_VALUE: p-paramter for the batch of ARIMA(p,q,d) models
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the memory where the new params are to be stored, which is also where the initial mapped input is stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, int P_VALUE, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void ar_param_invtransform_kernel(DataT *newParams, IdxT batchSize) {

  arctanh<DataT> arctanh;
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

   DataT tmp[P_VALUE];
   DataT myNewParams[P_VALUE];

  if(modelIndex<batchSize){
  //populating the local memory with the global memory

    #pragma unroll
    for(int i = 0; i<P_VALUE; ++i){
      tmp[i] = newParams[modelIndex*P_VALUE + i];
      myNewParams[i] = tmp[i];
    }

    for(int j=P_VALUE-1; j>0; --j){

      DataT a = myNewParams[j];

      for(int k = 0; k<j; ++k){

        tmp[k] = (myNewParams[k] + a*myNewParams[j-k-1])/(1-(a*a));
      }

      for(int iter = 0; iter<j; ++iter){

        myNewParams[iter] = tmp[iter];
      }

    }

    #pragma unroll
    for(int i = 0; i<P_VALUE; ++i){
      
      newParams[modelIndex*P_VALUE + i] =2*arctanh(myNewParams[i]);
    }
  }

}


/**
 * @brief kernel to perform jones inverse transformation on the moving average params
 * @tparam DataT: type of the params
 * @tparam Q_VALUE: q-paramter for the batch of ARIMA(p,d,q) models
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the memory where the new params are to be stored, which is also where the initial mapped input is stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, int Q_VALUE, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void ma_param_invtransform_kernel(DataT *newParams, IdxT batchSize) {

  arctanh<DataT> arctanh;
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

   DataT tmp[Q_VALUE];
   DataT myNewParams[Q_VALUE];

  if(modelIndex<batchSize){
  //populating the local memory with the global memory

    #pragma unroll
    for(int i = 0; i<Q_VALUE; ++i){
      tmp[i] = newParams[modelIndex*Q_VALUE + i];
      myNewParams[i] = tmp[i];
    }

    for(int j=Q_VALUE-1; j>0; --j){

      DataT b = myNewParams[j];

      for(int k = 0; k<j; ++k){

        tmp[k] = (myNewParams[k] - b*myNewParams[j-k-1])/(1-(b*b));
      }

      for(int iter = 0; iter<j; ++iter){

        myNewParams[iter] = tmp[iter];
      }

    }

    #pragma unroll
    for(int i = 0; i<Q_VALUE; ++i){

      newParams[modelIndex*Q_VALUE + i] = 2*arctanh(myNewParams[i]);
    }
  }

}

/**
 * @brief kernel to perform jones transformation on the autoregressive params
 * @tparam DataT: type of the params
 * @tparam P_VALUE: p-paramter for the batch of ARIMA(p,d,q) models
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the memory where the new params are to be stored, which is also where the initial mapped input is stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, int P_VALUE, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void ar_param_transform_kernel(DataT *newParams, IdxT batchSize) {

  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

   DataT tmp[P_VALUE];
   DataT myNewParams[P_VALUE];

  if(modelIndex<batchSize){
  //populating the local memory with the global memory

    #pragma unroll
    for(int i = 0; i<P_VALUE; ++i){
      tmp[i] = newParams[modelIndex*P_VALUE + i];
      myNewParams[i] = tmp[i];
    }

    for(int j=1; j<P_VALUE; ++j){

      DataT a = myNewParams[j];

      for(int k = 0; k<j; ++k){

        tmp[k] -= a* myNewParams[j-k-1];
      }

      for(int iter = 0; iter<j; ++iter){

        myNewParams[iter] = tmp[iter];
      }

    }

    #pragma unroll
    for(int i = 0; i<P_VALUE; ++i){
      newParams[modelIndex*P_VALUE + i] = myNewParams[i];
    }
  }

}

/**
 * @brief kernel to perform jones transformation on the moving average params
 * @tparam DataT: type of the params
 * @tparam Q_VALUE: q-paramter for the batch of ARIMA(p,d,q) models
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the memory where the new params are to be stored, which is also where the initial mapped input is stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, int Q_VALUE, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void ma_param_transform_kernel(DataT *newParams, IdxT batchSize) {

  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

   DataT tmp[Q_VALUE];
   DataT myNewParams[Q_VALUE];

  if(modelIndex<batchSize){
  //populating the local memory with the global memory
    #pragma unroll
    for(int i = 0; i<Q_VALUE; ++i){
      tmp[i] = newParams[modelIndex*Q_VALUE + i];
      myNewParams[i] = tmp[i];
    }

    for(int j=1; j<Q_VALUE; ++j){

      DataT a = myNewParams[j];

      for(int k = 0; k<j; ++k){

        tmp[k] += a* myNewParams[j-k-1];
      }

      #pragma unroll
      for(int iter = 0; iter<j; ++iter){

        myNewParams[iter] = tmp[iter];
      }

    }


    for(int i = 0; i<Q_VALUE; ++i){
      newParams[modelIndex*Q_VALUE + i] = myNewParams[i];
    }
  }

}


/**
* @brief Host Function to batchwise transform the autoregressive coefficients according to "jone's (1980)" transformation 
*
* @param params: 2D array where each row represents the AR coefficients of a particular model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param pValue: the number of coefficients per model (basically number of columns in params)
* @param newParams: the transformed params (output)
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT=int>
void ar_param_transform(
  const DataT* params, IdxT batchSize, IdxT pValue,
  DataT* newParams, std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {

  //rand index for size less than 2 is not defined
  ASSERT( batchSize>= 1 && pValue>=1, "not defined!");

  IdxT nElements = batchSize*pValue;

  //elementWise transforming the params matrix
  LinAlg::unaryOp(newParams, params, nElements, PAC<DataT>(), stream);

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x),1);

  //calling the kernel
  switch(pValue){

    case 1: ar_param_transform_kernel<DataT, 1, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 2: ar_param_transform_kernel<DataT, 2, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 3: ar_param_transform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 4: ar_param_transform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    default: ASSERT(false, "Unsupported pValue '%d'!", pValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}


/**
* @brief Host Function to batchwise transform the autoregressive coefficients according to "jone's (1980)" transformation 
*
* @param params: 2D array where each row represents the AR coefficients of a particular model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param pValue: the number of coefficients per model (basically number of columns in params)
* @param newParams: the transformed params (output)
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT=int>
void ar_param_inverse_transform(
  const DataT* params, IdxT batchSize, IdxT pValue,
  DataT* newParams, std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {

  //rand index for size less than 2 is not defined
  ASSERT( batchSize>= 1 && pValue>=1, "not defined!");

  IdxT nElements = batchSize*pValue;

  //elementWise transforming the params matrix
  copy(newParams, params, (size_t)nElements, stream);

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x),1);

  //calling the kernel
  switch(pValue){

    case 1: ar_param_invtransform_kernel<DataT, 1, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 2: ar_param_invtransform_kernel<DataT, 2, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 3: ar_param_invtransform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 4: ar_param_invtransform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    default: ASSERT(false, "Unsupported pValue '%d'!", pValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}


/**
* @brief Host Function to batchwise transform the moving average coefficients according to "jone's (1980)" transformation 
*
* @param params: 2D array where each row represents the MA coefficients of a particular model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param qValue: the number of coefficients per model (basically number of columns in params)
* @param newParams: the transformed params (output)
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT=int>
void ma_param_transform(
  const DataT* params, IdxT batchSize, IdxT qValue,
  DataT* newParams, std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {
  //rand index for size less than 2 is not defined
  ASSERT( batchSize>= 1 && qValue>=1, "not defined!");

  IdxT nElements = batchSize*qValue;

  //elementWise transforming the params matrix
  LinAlg::unaryOp(newParams, params, nElements, PAC<DataT>(), stream);

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x),1);

  //calling the kernel
  switch(qValue){

    case 1: ma_param_transform_kernel<DataT, 1, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 2: ma_param_transform_kernel<DataT, 2, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 3: ma_param_transform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 4: ma_param_transform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    default: ASSERT(false, "Unsupported qValue '%d'!", qValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}


/**
* @brief Host Function to batchwise inverse transform the moving average coefficients according to "jone's (1980)" transformation 
*
* @param params: 2D array where each row represents the transformed MA coefficients of a transformed model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param qValue: the number of coefficients per model (basically number of columns in params)
* @param newParams: the inverse transformed params (output)
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT=int>
void ma_param_inverse_transform(
  const DataT* params, IdxT batchSize, IdxT qValue,
  DataT* newParams, std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {

  //rand index for size less than 2 is not defined
  ASSERT( batchSize>= 1 && qValue>=1, "not defined!");

  IdxT nElements = batchSize*qValue;

//copying contents
  copy(newParams, params, (size_t)nElements, stream);

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x),1);

  //calling the kernel
  switch(qValue){

    case 1: ma_param_invtransform_kernel<DataT, 1, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 2: ma_param_invtransform_kernel<DataT, 2, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 3: ma_param_invtransform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    case 4: ma_param_invtransform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
              <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
              break;
    default: ASSERT(false, "Unsupported qValue '%d'!", qValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}
  

};  //end namespace TimeSeries
};  //end namespace MLCommon
