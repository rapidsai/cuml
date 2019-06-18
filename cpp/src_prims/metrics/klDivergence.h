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
* @file klDivergence.h
* @brief The KL divergence tells us how well the probability distribution Q AKA candidatePDF 
* approximates the probability distribution P AKA modelPDF.
*/

#include <math.h>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/map_then_reduce.h"

namespace MLCommon {

/**
* @brief the KL Diverence mapping function
*
* @tparam Type: Data type of the input 
* @param modelPDF: the model probability density function of type DataT
* @param candidatePDF: the candidate probability density function of type DataT
*/
template <typename Type>
struct KLDOp {
  HDI Type operator()(Type modelPDF, Type candidatePDF) {
    if (modelPDF == 0.0)
      return 0;

    else
      return modelPDF * (log(modelPDF) - log(candidatePDF));
  }
};

namespace Metrics {

/**
* @brief Function to calculate KL Divergence
* <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL Divergence</a> 
* @param modelPDF: the model array of probability density functions of type DataT
* @param candidatePDF: the candidate array of probability density functions of type DataT
* @param size: the size of the data points of type int
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename DataT>
DataT klDivergence(const DataT* modelPDF, const DataT* candidatePDF, int size,
                   std::shared_ptr<MLCommon::deviceAllocator> allocator,
                   cudaStream_t stream) {
  //DataT epsilon = 0.000001;
  // auto addEpsilon = [&](DataT &el){ el+= epsilon; };

  MLCommon::device_buffer<DataT> modelPDFCopy(allocator, stream, size);
  MLCommon::device_buffer<DataT> candidatePDFCopy(allocator, stream, size);

  MLCommon::copy(modelPDFCopy.data(), modelPDF, size, stream);
  MLCommon::copy(candidatePDFCopy.data(), candidatePDF, size, stream);

  //std::for_each(modelPDFCopy.data(), modelPDFCopy.data()+size, addEpsilon);
  //std::for_each(candidatePDFCopy.data(), candidatePDFCopy.data()+size, addEpsilon);

  MLCommon::device_buffer<DataT> d_KLDVal(allocator, stream, 1);
  CUDA_CHECK(cudaMemsetAsync(d_KLDVal.data(), 0, sizeof(DataT), stream));

  MLCommon::LinAlg::mapThenSumReduce<DataT, KLDOp<DataT>, 256, DataT*>(
    d_KLDVal.data(), (size_t)size, KLDOp<DataT>(), stream, modelPDFCopy.data(),
    candidatePDFCopy.data());

  DataT h_KLDVal;

  MLCommon::updateHost(&h_KLDVal, d_KLDVal.data(), 1, stream);

  return h_KLDVal;
}

};  //end namespace Metrics
};  //end namespace MLCommon
