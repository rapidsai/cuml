/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
 * @file jones_transform.cuh
 * @brief Transforms params to induce stationarity/invertability.
 * reference: Jones(1980)
 */

#pragma once

#include <cuml/common/utils.hpp>

#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <math.h>

namespace MLCommon {

namespace TimeSeries {

/**
* @brief Inline device function for the transformation operation

* @tparam Type: Data type of the input
* @param tmp: the temporary array used in transformation
* @param myNewParams: will contain the transformed params
* @param parameter: the pValue/qValue for the transformation
* @param isAr: tell the type of transform (if ar or ma transform)
* @param clamp: whether to clamp transformed params between -1 and 1
*/
template <typename DataT>
DI void transform(DataT* tmp, DataT* myNewParams, int parameter, bool isAr, bool clamp)
{
  for (int i = 0; i < parameter; ++i) {
    tmp[i]         = raft::tanh(tmp[i] * 0.5);
    myNewParams[i] = tmp[i];
  }

  DataT sign = isAr ? -1 : 1;
  for (int j = 1; j < parameter; ++j) {
    DataT a = myNewParams[j];
    for (int k = 0; k < j; ++k) {
      tmp[k] += sign * (a * myNewParams[j - k - 1]);
    }
    for (int iter = 0; iter < j; ++iter) {
      myNewParams[iter] = tmp[iter];
    }
  }

  if (clamp) {
    // Clamp values to avoid numerical issues when very close to 1
    for (int i = 0; i < parameter; ++i) {
      myNewParams[i] = max(-0.9999, min(myNewParams[i], 0.9999));
    }
  }
}

/**
* @brief Inline device function for the inverse transformation operation

* @tparam Type: Data type of the input
* @param tmp: the temporary array used in transformation
* @param myNewParams: will contain the transformed params
* @param parameter: the pValue/qValue for the inverse transformation
* @param isAr: tell the type of inverse transform (if ar or ma transform)
*/
template <typename DataT>
DI void invtransform(DataT* tmp, DataT* myNewParams, int parameter, bool isAr)
{
  DataT sign = isAr ? 1 : -1;
  for (int j = parameter - 1; j > 0; --j) {
    DataT a = myNewParams[j];

    for (int k = 0; k < j; ++k) {
      tmp[k] = (myNewParams[k] + sign * (a * myNewParams[j - k - 1])) / (1 - (a * a));
    }

    for (int iter = 0; iter < j; ++iter) {
      myNewParams[iter] = tmp[iter];
    }
  }

  for (int i = 0; i < parameter; ++i) {
    myNewParams[i] = 2 * raft::atanh(myNewParams[i]);
  }
}

/**
 * @brief kernel to perform jones transformation
 * @tparam DataT: type of the params
 * @param newParams: pointer to the memory where the new params are to be stored
 * @param params: pointer to the memory where the initial params are stored
 * @param batchSize: number of models in a batch
 * @param parameter: the parameter for the batch of ARIMA(p,q,d) models (either p or q
 * depending on whether coefficients are of type AR or MA respectively)
 * @param isAr: if the coefficients to be transformed are Autoregressive or moving average
 * @param isInv: if the transformation type is regular or inverse
 * @param clamp: whether to clamp transformed params between -1 and 1
 */
template <typename DataT>
CUML_KERNEL void jones_transform_kernel(DataT* newParams,
                                        const DataT* params,
                                        int batchSize,
                                        int parameter,
                                        bool isAr,
                                        bool isInv,
                                        bool clamp)
{
  // calculating the index of the model that the coefficients belong to
  int modelIndex = threadIdx.x + ((int)blockIdx.x * blockDim.x);

  DataT tmp[8];
  DataT myNewParams[8];

  if (modelIndex < batchSize) {
    // load
    for (int i = 0; i < parameter; ++i) {
      tmp[i]         = params[modelIndex * parameter + i];
      myNewParams[i] = tmp[i];
    }

    // the transformation/inverse transformation operation
    if (isInv)
      invtransform<DataT>(tmp, myNewParams, parameter, isAr);
    else
      transform<DataT>(tmp, myNewParams, parameter, isAr, clamp);

    // store
    for (int i = 0; i < parameter; ++i) {
      newParams[modelIndex * parameter + i] = myNewParams[i];
    }
  }
}

/**
 * @brief Host Function to batchwise transform/inverse transform the moving average
 * coefficients/autoregressive coefficients according to "jone's (1980)" transformation
 *
 * @param params: 2D array where each row represents the transformed MA coefficients of a
 * transformed model
 * @param batchSize: the number of models in a batch (number of rows in params)
 * @param parameter: the number of coefficients per model (basically number of columns in params)
 * @param newParams: the inverse transformed params (output)
 * @param isAr: set to true if the params to be transformed are Autoregressive params, false if
 * params are of type MA
 * @param isInv: set to true if the transformation is an inverse type transformation, false if
 * regular transform
 * @param stream: the cudaStream object
 * @param clamp: whether to clamp transformed params between -1 and 1
 */
template <typename DataT>
void jones_transform(const DataT* params,
                     int batchSize,
                     int parameter,
                     DataT* newParams,
                     bool isAr,
                     bool isInv,
                     cudaStream_t stream,
                     bool clamp = true)
{
  ASSERT(batchSize >= 1, "Unsupported batchSize '%d'!", batchSize);
  ASSERT(parameter >= 1 && parameter <= 8, "Unsupported parameter '%d'!", parameter);

  int nElements = batchSize * parameter;

  // copying contents
  raft::copy(newParams, params, (size_t)nElements, stream);

  // setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(raft::ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);

  jones_transform_kernel<DataT><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
    newParams, params, batchSize, parameter, isAr, isInv, clamp);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

};  // end namespace TimeSeries
};  // end namespace MLCommon
