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
#include "vectorized.h"

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
* @brief Lambda to map to the inverse partial autocorrelation
*
* @tparam Type: Data type of the input 
* @param in: the input to the functional mapping
* @return : the inverse Partial autocorrelation (ie, 2*arctanh(in))
*/
template <typename Type>
struct invPAC {
  HDI Type operator()(Type in) { return 2.0 * myATanh(in); }
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
}

/**
 * @brief kernel to perform jones inverse transformation on the autoregressive params for p=4
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p4_ar_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[4];
  DataT myNewParams[4];
  TxN_t<DataT, 2> vectorized_tmp;

  if (globalIndex < 2 * batchSize) {
    //populating the local memory with the global memory
    //load
    vectorized_tmp.load(params, globalIndex * 2);

    if (globalIndex % 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i + 2] = vectorized_tmp.val.data[i];
        myNewParams[i + 2] = tmp[i + 2];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i] = vectorized_tmp.val.data[i];
        myNewParams[i] = tmp[i];
      }
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //inverse transformation
    if (globalIndex % 2) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];

      invtransform<DataT, IdxT, 4>(tmp, myNewParams, true);
    }

    //store
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2 == 0) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i + 2];
      }
    }

    vectorized_tmp.val.data[0] = 2.0 * myATanh(vectorized_tmp.val.data[0]);
    vectorized_tmp.val.data[1] = 2.0 * myATanh(vectorized_tmp.val.data[1]);
    vectorized_tmp.store(newParams, globalIndex * 2);
  }
}

/**
 * @brief kernel to perform jones inverse transformation on the autoregressive params for p=3
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p3_ar_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[3];
  DataT myNewParams[3];

  if (globalIndex < batchSize) {
    //populating the local memory with the global memory

    //load the elements
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      tmp[i] = params[3 * globalIndex + i];
      myNewParams[i] = tmp[i];
    }

    //the inverse transforamtion
    invtransform<DataT, IdxT, 3>(tmp, myNewParams, true);

//applying the arctanh mapping and storing to global memory
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      newParams[globalIndex * 3 + i] = 2.0 * myATanh(myNewParams[i]);
    }
  }
}

/**
 * @brief kernel to perform jones inverse transformation on the autoregressive params for p=2
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p2_ar_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[2];
  DataT myNewParams[2];
  TxN_t<DataT, 2> vectorized_tmp;

  if (modelIndex < batchSize) {
    //load

    vectorized_tmp.load(params, 2 * modelIndex);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmp[i] = vectorized_tmp.val.data[i];
      myNewParams[i] = tmp[i];
    }

    //inverse transformation

    invtransform<DataT, IdxT, 2>(tmp, myNewParams, true);

    //applying the arctanh mapping and storing

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      vectorized_tmp.val.data[i] = 2.0 * myATanh(myNewParams[i]);
    }

    vectorized_tmp.store(newParams, 2 * modelIndex);
  }
}

/**
 * @brief kernel to perform jones inverse transformation on the moving average params for q=4
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p4_ma_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[4];
  DataT myNewParams[4];
  TxN_t<DataT, 2> vectorized_tmp;

  if (globalIndex < 2 * batchSize) {
    //populating the local memory with the global memory
    //load
    vectorized_tmp.load(params, globalIndex * 2);

    if (globalIndex % 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i + 2] = vectorized_tmp.val.data[i];
        myNewParams[i + 2] = tmp[i + 2];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i] = vectorized_tmp.val.data[i];
        myNewParams[i] = tmp[i];
      }
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //inverse transformation
    if (globalIndex % 2) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];

      invtransform<DataT, IdxT, 4>(tmp, myNewParams, false);
    }

    //store
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2 == 0) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i + 2];
      }
    }

    vectorized_tmp.val.data[0] = 2.0 * myATanh(vectorized_tmp.val.data[0]);
    vectorized_tmp.val.data[1] = 2.0 * myATanh(vectorized_tmp.val.data[1]);
    vectorized_tmp.store(newParams, globalIndex * 2);
  }
}

/**
 * @brief kernel to perform jones inverse transformation on the moving average params for q=3
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p3_ma_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[3];
  DataT myNewParams[3];

  if (globalIndex < batchSize) {
    //populating the local memory with the global memory

    //load the elements
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      tmp[i] = params[3 * globalIndex + i];
      myNewParams[i] = tmp[i];
    }

    //the inverse transforamtion
    invtransform<DataT, IdxT, 3>(tmp, myNewParams, false);

//applying the arctanh mapping and storing to global memory
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      newParams[globalIndex * 3 + i] = 2.0 * myATanh(myNewParams[i]);
    }
  }
}

/**
 * @brief kernel to perform jones inverse transformation on the moving average params for q=2
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p2_ma_param_invtransform_kernel(DataT* newParams,
                                                const DataT* params,
                                                IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);

  DataT tmp[2];
  DataT myNewParams[2];
  TxN_t<DataT, 2> vectorized_tmp;

  if (modelIndex < batchSize) {
    //load

    vectorized_tmp.load(params, 2 * modelIndex);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmp[i] = vectorized_tmp.val.data[i];
      myNewParams[i] = tmp[i];
    }

    //inverse transformation
    invtransform<DataT, IdxT, 2>(tmp, myNewParams, false);

    //applying the arctanh mapping and storing

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      vectorized_tmp.val.data[i] = 2.0 * myATanh(myNewParams[i]);
    }

    vectorized_tmp.store(newParams, 2 * modelIndex);
  }
}

/**
 * @brief kernel to perform jones transformation on autoregressive params for p=2
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p2_ar_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[2];
  DataT myNewParams[2];
  TxN_t<DataT, 2> vectorized_tmp;

  if (modelIndex < batchSize) {
    //loading from global memory
    vectorized_tmp.load(params, modelIndex * 2);
//load
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmp[i] = pac(vectorized_tmp.val.data[i]);
      myNewParams[i] = tmp[i];
    }

    //transform
    transform<DataT, IdxT, 2>(tmp, myNewParams, true);

//store
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      vectorized_tmp.val.data[i] = myNewParams[i];
    }
    //storing in global memory
    vectorized_tmp.store(newParams, modelIndex * 2);
  }
}

/**
 * @brief kernel to perform jones transformation on autoregressive params for p=4
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p4_ar_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[4];
  DataT myNewParams[4];
  TxN_t<DataT, 2> vectorized_tmp;

  if (globalIndex < 2 * batchSize) {
    //load
    vectorized_tmp.load(params, globalIndex * (2));

    if (globalIndex % 2) {
      tmp[2] = pac(vectorized_tmp.val.data[0]);
      myNewParams[2] = tmp[2];

      tmp[3] = pac(vectorized_tmp.val.data[1]);
      myNewParams[3] = tmp[3];

    } else {
      tmp[0] = pac(vectorized_tmp.val.data[0]);
      myNewParams[0] = tmp[0];

      tmp[1] = pac(vectorized_tmp.val.data[1]);
      myNewParams[1] = tmp[1];
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //the odd thread does the required operation

    if (globalIndex % 2 == 1) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];

      transform<DataT, IdxT, 4>(tmp, myNewParams, true);
    }

    //storing part
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2) {
      vectorized_tmp.val.data[0] = myNewParams[2];
      vectorized_tmp.val.data[1] = myNewParams[3];

    } else {
      vectorized_tmp.val.data[0] = myNewParams[0];
      vectorized_tmp.val.data[1] = myNewParams[1];
    }

    vectorized_tmp.store(newParams, globalIndex * (2));
  }
}

/**
 * @brief kernel to perform jones transformation on autoregressive params for p=3
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void p3_ar_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[3];
  DataT myNewParams[3];

  if (globalIndex < 2 * batchSize) {
    //load

    if (globalIndex % 2) {
      tmp[2] = pac(params[(3 * globalIndex + 1) / 2 + 0]);
      myNewParams[2] = tmp[2];
    } else {
      tmp[0] = pac(params[(3 * globalIndex + 1) / 2 + 0]);
      myNewParams[0] = tmp[0];

      tmp[1] = pac(params[(3 * globalIndex + 1) / 2 + 1]);
      myNewParams[1] = tmp[1];
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //the odd thread does the required operation

    if (globalIndex % 2 == 1) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];
      transform<DataT, IdxT, 3>(tmp, myNewParams, true);
    }

    //storing part
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2) {
      newParams[(3 * globalIndex + 1) / 2 + 0] = myNewParams[2];

    } else {
      newParams[(3 * globalIndex + 1) / 2 + 0] = myNewParams[0];
      newParams[(3 * globalIndex + 1) / 2 + 1] = myNewParams[1];
    }
  }
}

/**
 * @brief kernel to perform jones transformation on moving average params for q=4
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void q4_ma_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[4];
  DataT myNewParams[4];
  TxN_t<DataT, 2> vectorized_tmp;

  if (globalIndex < 2 * batchSize) {
    //load
    vectorized_tmp.load(params, 2 * globalIndex);
    if (globalIndex % 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i + 2] = pac(vectorized_tmp.val.data[i]);
        myNewParams[i + 2] = tmp[i + 2];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        tmp[i] = pac(vectorized_tmp.val.data[i]);
        myNewParams[i] = tmp[i];
      }
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //the odd thread does the required operation

    if (globalIndex % 2 == 1) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];
      transform<DataT, IdxT, 4>(tmp, myNewParams, false);
    }

    //storing part
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i + 2];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        vectorized_tmp.val.data[i] = myNewParams[i];
      }
    }

    //storing to the global memory
    vectorized_tmp.store(newParams, globalIndex * 2);
  }
}

/**
 * @brief kernel to perform jones transformation on moving average params for q=3
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void q3_ma_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT globalIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[3];
  DataT myNewParams[3];

  if (globalIndex < 2 * batchSize) {
    //load

    if (globalIndex % 2) {
      tmp[2] = pac(params[(3 * globalIndex + 1) / 2 + 0]);
      myNewParams[2] = tmp[2];
    } else {
      tmp[0] = pac(params[(3 * globalIndex + 1) / 2 + 0]);
      myNewParams[0] = tmp[0];

      tmp[1] = pac(params[(3 * globalIndex + 1) / 2 + 1]);
      myNewParams[1] = tmp[1];
    }

    tmp[0] = __shfl_up_sync(0xffffffff, tmp[0], 1);
    tmp[1] = __shfl_up_sync(0xffffffff, tmp[1], 1);

    //the odd thread does the required operation

    if (globalIndex % 2 == 1) {
      myNewParams[0] = tmp[0];
      myNewParams[1] = tmp[1];
      transform<DataT, IdxT, 3>(tmp, myNewParams, false);
    }

    //storing part
    myNewParams[0] = __shfl_down_sync(0xffffffff, myNewParams[0], 1);
    myNewParams[1] = __shfl_down_sync(0xffffffff, myNewParams[1], 1);

    if (globalIndex % 2) {
      newParams[(3 * globalIndex + 1) / 2 + 0] = myNewParams[2];

    } else {
      newParams[(3 * globalIndex + 1) / 2 + 0] = myNewParams[0];
      newParams[(3 * globalIndex + 1) / 2 + 1] = myNewParams[1];
    }
  }
}

/**
 * @brief kernel to perform jones transformation on moving average params for q=2
 * @tparam DataT: type of the params
 * @tparam IdxT: type of indexing
 * @tparam BLOCK_DIM_X: number of threads in block in x dimension
 * @tparam BLOCK_DIM_Y: number of threads in block in y dimension
 * @param newParams: pointer to the location where transformed parameters are stored
 * @param newParams: pointer to the location where the initial parameters are stored
 * @param batchSize: number of models in a batch
 */
template <typename DataT, typename IdxT, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void q2_ma_param_transform_kernel(DataT* newParams,
                                             const DataT* params,
                                             IdxT batchSize) {
  //calculating the index of the model that the coefficients belong to
  IdxT modelIndex = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  PAC<DataT> pac;
  DataT tmp[2];
  DataT myNewParams[2];
  TxN_t<DataT, 2> vectorized_tmp;

  if (modelIndex < batchSize) {
    //loading the global memory and performing the PAC mapping
    vectorized_tmp.load(params, modelIndex * 2);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmp[i] = pac(vectorized_tmp.val.data[i]);
      myNewParams[i] = tmp[i];
    }

    //the transformation
    transform<DataT, IdxT, 2>(tmp, myNewParams, false);

    //storing
    for (int i = 0; i < 2; ++i) {
      vectorized_tmp.val.data[i] = myNewParams[i];
    }

    vectorized_tmp.store(newParams, modelIndex * 2);
  }
}

/**
* @brief Host Function to batchwise transform the autoregressive coefficients according to "jone's (1980)" transformation 
*
* @tparam DataT: the data type of input
* @tparam: IdxT: index data type
* @param params: 2D array where each row represents the AR coefficients of a particular model
* @param batchSize: the number of models in a batch (number of rows in params)
* @param pValue: the number of coefficients per model (basically number of columns in params)
* @param newParams: the transformed params (output)
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT, typename IdxT = int>
void ar_param_transform(const DataT* params, IdxT batchSize, IdxT pValue,
                        DataT* newParams,
                        std::shared_ptr<MLCommon::deviceAllocator> allocator,
                        cudaStream_t stream) {
  ASSERT(batchSize >= 1 && pValue >= 1, "not defined!");

  IdxT nElements = batchSize * pValue;

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 p2numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);
  dim3 p4numBlocks(ceildiv<int>(2 * batchSize, numThreadsPerBlock.x), 1);
  dim3 p3numBlocks(ceildiv<int>(2 * batchSize, numThreadsPerBlock.x), 1);

  //calling the kernel
  switch (pValue) {
    case 1:
      LinAlg::unaryOp(newParams, params, nElements, PAC<DataT>(), stream);
      break;
    case 2:
      p2_ar_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p2numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    case 3:
      //ar_param_transform_kernel<DataT, 3, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
      //<<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
      p3_ar_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p3numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    case 4:
      //ar_param_transform_kernel<DataT, 4, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
      //<<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, batchSize);
      p4_ar_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p4numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    default:
      ASSERT(false, "Unsupported pValue '%d'!", pValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());
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
template <typename DataT, typename IdxT = int>
void ar_param_inverse_transform(
  const DataT* params, IdxT batchSize, IdxT pValue, DataT* newParams,
  std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {
  ASSERT(batchSize >= 1 && pValue >= 1, "not defined!");

  IdxT nElements = batchSize * pValue;

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);
  dim3 p4numBlocks(ceildiv<int>(2 * batchSize, numThreadsPerBlock.x), 1);

  //calling the kernel
  switch (pValue) {
    case 1:
      LinAlg::unaryOp(newParams, params, nElements, invPAC<DataT>(), stream);
      break;
    case 2:
      p2_ar_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize);
      break;
    case 3:
      p3_ar_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize);
      break;
    case 4:
      p4_ar_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p4numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    default:
      ASSERT(false, "Unsupported pValue '%d'!", pValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());
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
template <typename DataT, typename IdxT = int>
void ma_param_transform(const DataT* params, IdxT batchSize, IdxT qValue,
                        DataT* newParams,
                        std::shared_ptr<MLCommon::deviceAllocator> allocator,
                        cudaStream_t stream) {
  ASSERT(batchSize >= 1 && qValue >= 1, "not defined!");

  IdxT nElements = batchSize * qValue;

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);
  dim3 p4numBlocks(ceildiv<int>(2 * batchSize, numThreadsPerBlock.x), 1);

  //calling the kernel
  switch (qValue) {
    case 1:
      LinAlg::unaryOp(newParams, params, nElements, PAC<DataT>(), stream);
      break;
    case 2:
      q2_ma_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize);
      break;
    case 3:
      q3_ma_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p4numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    case 4:
      q4_ma_param_transform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<p4numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    default:
      ASSERT(false, "Unsupported qValue '%d'!", qValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());
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
template <typename DataT, typename IdxT = int>
void ma_param_inverse_transform(
  const DataT* params, IdxT batchSize, IdxT qValue, DataT* newParams,
  std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {
  ASSERT(batchSize >= 1 && qValue >= 1, "not defined!");

  IdxT nElements = batchSize * qValue;

  //setting the kernel configuration
  static const int BLOCK_DIM_Y = 1, BLOCK_DIM_X = 256;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(batchSize, numThreadsPerBlock.x), 1);
  dim3 q4numBlocks(ceildiv<int>(2 * batchSize, numThreadsPerBlock.x), 1);

  //calling the kernel
  switch (qValue) {
    case 1:
      LinAlg::unaryOp(newParams, params, nElements, invPAC<DataT>(), stream);
      break;
    case 2:
      p2_ma_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize);
      break;
    case 3:
      p3_ma_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                       batchSize);
      break;
    case 4:
      p4_ma_param_invtransform_kernel<DataT, IdxT, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<q4numBlocks, numThreadsPerBlock, 0, stream>>>(newParams, params,
                                                         batchSize);
      break;
    default:
      ASSERT(false, "Unsupported pValue '%d'!", qValue);
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace TimeSeries
};  //end namespace MLCommon
