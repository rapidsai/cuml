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
* @file jones_transform.h
* @brief Transforms params to induce stationarity/invertability.
* reference: Jones(1980) 

*/

#include <math.h>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/unary_op.h"

namespace MLCommon {

namespace TimeSeries {

/**
* @brief Lambda to map to the partial autocorrelation
*
* @tparam Type: Data type of the input 
* @param in: the input to the functional mapping
* @return : the Partial autocorrelation (ie, tanh(in/2))
*/
template <typename Type>
struct PAC {
  HDI Type operator()(Type in) { return myTanh(in * 0.5); }
};

/**
* @brief Inline device function for the transformation operation

* @tparam Type: Data type of the input
* @tparam IdxT: indexing data type
* @tparam value: the pValue/qValue for the transformation
* @param tmp: the temporary array used in transformation
* @param myNewParam: will contain the transformed params
* @param: isAr: tell the type of transform (if ar or ma transform)
*/
template <typename DataT, typename IdxT, int VALUE>
inline __device__ void transform(DataT* tmp, DataT* myNewParams, bool isAr) {
  //do the ar transformation
  PAC<DataT> pac;
  for (int i = 0; i < VALUE; ++i) {
    tmp[i] = pac(tmp[i]);
    myNewParams[i] = tmp[i];
  }
  if (isAr) {
    for (int j = 1; j < VALUE; ++j) {
      DataT a = myNewParams[j];

      for (int k = 0; k < j; ++k) {
        tmp[k] -= a * myNewParams[j - k - 1];
      }

      for (int iter = 0; iter < j; ++iter) {
        myNewParams[iter] = tmp[iter];
      }
    }
  } else {  //do the ma transformation
    for (int j = 1; j < VALUE; ++j) {
      DataT a = myNewParams[j];

      for (int k = 0; k < j; ++k) {
        tmp[k] += a * myNewParams[j - k - 1];
      }

      for (int iter = 0; iter < j; ++iter) {
        myNewParams[iter] = tmp[iter];
      }
    }
  }
}

/**
* @brief Inline device function for the inverse transformation operation

* @tparam Type: Data type of the input
* @tparam IdxT: indexing data type
* @tparam value: the pValue/qValue for the inverse transformation
* @param tmp: the temporary array used in transformation
* @param myNewParam: will contain the transformed params
* @param: isAr: tell the type of inverse transform (if ar or ma transform)
*/
template <typename DataT, typename IdxT, int VALUE>
inline __device__ void invtransform(DataT* tmp, DataT* myNewParams, bool isAr) {
  //do the ar transformation
  if (isAr) {
    for (int j = VALUE - 1; j > 0; --j) {
      DataT a = myNewParams[j];

      for (int k = 0; k < j; ++k) {
        tmp[k] = (myNewParams[k] + a * myNewParams[j - k - 1]) / (1 - (a * a));
      }

      for (int iter = 0; iter < j; ++iter) {
        myNewParams[iter] = tmp[iter];
      }
    }
  } else {  //do the ma transformation
    for (int j = VALUE - 1; j > 0; --j) {
      DataT a = myNewParams[j];

      for (int k = 0; k < j; ++k) {
        tmp[k] = (myNewParams[k] - a * myNewParams[j - k - 1]) / (1 - (a * a));
      }

      for (int iter = 0; iter < j; ++iter) {
        myNewParams[iter] = tmp[iter];
      }
    }
  }

  for (int i = 0; i < VALUE; ++i) {
    myNewParams[i] = 2 * myATanh(myNewParams[i]);
  }
}

/**
 * @brief kernel to perform jones transformation
 * @tparam DataT: type of the params
 * @tparam VALUE: the parameter for the batch of ARIMA(p,q,d) models (either p or q depending on whether coefficients are of type AR or MA respectively)
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the memory where the new params are to be stored
 * @param params: pointer to the memory where the initial params are stored
 * @param batchSize: number of models in a batch
 * @param isAr: if the coefficients to be transformed are Autoregressive or moving average
 * @param isInv: if the transformation type is regular or inverse 
 */
template <typename DataT, int VALUE, typename IdxT, int BLOCK_DIM_X,
          int BLOCK_DIM_Y>
__global__ void jones_transform_kernel(DataT* newParams, const DataT* params,
                                       IdxT batchSize, bool isAr, bool isInv) {
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[VALUE];
  DataT myNewParams[VALUE];

  if (modelIndex < batchSize) {
//load
#pragma unroll
    for (int i = 0; i < VALUE; ++i) {
      tmp[i] = params[modelIndex * VALUE + i];
      myNewParams[i] = tmp[i];
    }

    //the transformation/inverse transformation operation
    if (isInv)
      invtransform<DataT, IdxT, VALUE>(tmp, myNewParams, isAr);
    else
      transform<DataT, IdxT, VALUE>(tmp, myNewParams, isAr);

//store
#pragma unroll
    for (int i = 0; i < VALUE; ++i) {
      newParams[modelIndex * VALUE + i] = myNewParams[i];
    }
  }
}

/**
* @brief Host Function to batchwise transform/inverse transform the moving average coefficients/autoregressive coefficients according to "jone's (1980)" transformation 
*
* @param params: 2D array where each row represents the transformed MA coefficients of a transformed model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param parameter: the number of coefficients per model (basically number of columns in params)
* @param newParams: the inverse transformed params (output)
* @param isAR: set to true if the params to be transformed are Autoregressive params, false if params are of type MA
* @param isInv: set to true if the transformation is an inverse type transformation, false if regular transform
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT = int>
void jones_transform(const DataT* params, IdxT batchSize, IdxT parameter,
                     DataT* newParams, bool isAr, bool isInv,
                     std::shared_ptr<MLCommon::deviceAllocator> allocator,
                     cudaStream_t stream) {
  ASSERT(batchSize >= 1 && parameter >= 1, "not defined!");

  IdxT nElements = batchSize * parameter;

  //copying contents
  copy(newParams, params, (size_t)nElements, stream);

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);

  //calling the kernel

  switch (parameter) {
    case 1:
      jones_transform_kernel<DataT, 1, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize, isAr, isInv);
      break;
    case 2:
      jones_transform_kernel<DataT, 2, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize, isAr, isInv);
      break;
    case 3:
      jones_transform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize, isAr, isInv);
      break;
    case 4:
      jones_transform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize, isAr, isInv);
      break;
    default:
      ASSERT(false, "Unsupported parameter '%d'!", parameter);
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

};  //end namespace TimeSeries
};  //end namespace MLCommon
