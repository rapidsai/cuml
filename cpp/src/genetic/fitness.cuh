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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/mean_squared_error.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>

#include <raft/cudart_utils.h>
#include <rmm/device_uvector.hpp>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <stats/weighted_mean.cuh>
#include <functions/logisticReg.cuh>

namespace cuml {
namespace genetic {

template <typename math_t>
void _weighted_pearson(cudaStream_t stream, int len, math_t* y, math_t* y_pred, math_t* w,math_t* out) {

  rmm::device_uvector<math_t> mu_y(1,stream);
  rmm::device_uvector<math_t> mu_y_pred(1,stream);
  rmm::device_uvector<math_t> stddev_y(1,stream);
  rmm::device_uvector<math_t> stddev_ypred(1,stream);
  rmm::device_uvector<math_t> y_centred(len,stream);
  rmm::device_uvector<math_t> y_pred_centred(len,stream);
  rmm::device_uvector<math_t> ele_covariance(len,stream);
  rmm::device_uvector<math_t> sample_covariance(1,stream);
  
  MLCommon::Stats::colWeightedMean(mu_y.data(),y, w, 1, len, stream);
  raft::stats::meanCenter(y_centred.data(),y,mu_y.data(),1,len,false,true,stream);
  raft::stats::stddev(stddev_y.data(),y,mu_y.data(),1,len,true,false,stream);

  MLCommon::Stats::colWeightedMean(mu_y_pred.data(),y_pred, w, 1, len, stream);
  raft::stats::meanCenter(y_pred_centred.data(),y,mu_y_pred.data(),1,len,false,true,stream);
  raft::stats::stddev(stddev_ypred.data(),y,mu_y_pred.data(),1,len,true,false,stream);

  raft::linalg::eltwiseMultiply(ele_covariance.data(),y_centred.data(),y_pred_centred.data(), len,stream);
  MLCommon::Stats::colWeightedMean(sample_covariance.data(),ele_covariance.data(), w, 1, len, stream);

  math_t correlation = 0.0;
  correlation = sample_covariance.front_element(stream)/(stddev_y.front_element(stream)*stddev_ypred.front_element(stream));
  raft::update_device(out, &correlation, 1, stream);
}

template <typename math_t>
void _weighted_spearman(cudaStream_t stream,int len, math_t* y, math_t* y_pred, math_t* w, math_t* out) {
  
  // Find ranks of y and y_pred 
  rmm::device_uvector<math_t> copy_y(len,stream);
  rmm::device_uvector<math_t> temp_y(len,stream);
  rmm::device_uvector<math_t> rank_y(len,stream);
  
  rmm::device_uvector<math_t> copy_ypred(len,stream);
  rmm::device_uvector<math_t> temp_ypred(len,stream);
  rmm::device_uvector<math_t> rank_ypred(len,stream);
  
  thrust::sequence(thrust::cuda::par.on(stream),rank_y.begin(),rank_y.end(),1);
  thrust::sequence(thrust::cuda::par.on(stream),rank_ypred.begin(),rank_ypred.end(),1);
  thrust::sequence(thrust::cuda::par.on(stream),temp_y.begin(),temp_y.end(),1);
  thrust::sequence(thrust::cuda::par.on(stream),temp_ypred.begin(),temp_ypred.end(),1);

  raft::copy_async(copy_y.data(),y,len,stream);
  raft::copy_async(copy_ypred.data(),y_pred,len,stream);

  // Argsort y and y_pred
  thrust::sort_by_key(thrust::cuda::par.on(stream), copy_y.begin(),copy_y.end(),temp_y.begin());
  thrust::sort_by_key(thrust::cuda::par.on(stream), temp_y.begin(), temp_y.end(),rank_y.begin());

  thrust::sort_by_key(thrust::cuda::par.on(stream), copy_ypred.begin(),copy_ypred.end(),temp_ypred.begin());
  thrust::sort_by_key(thrust::cuda::par.on(stream), temp_ypred.begin(), temp_ypred.end(),rank_ypred.begin());
  
  // Find pearson coeff
  _weighted_pearson(stream, len, rank_y.data(), rank_ypred.data(), w, out);
  
}

template <typename math_t>
void _mean_absolute_error(cudaStream_t stream,int len, math_t* y, math_t* y_pred, math_t* w, math_t* out) {       
  // Find weighted absolute error
  rmm::device_uvector<math_t> error(len, stream);
  raft::linalg::subtract(error.data(), y, y_pred, len, stream);
  raft::linalg::unaryOp(error.data(), error.data(), len, 
    [] __device__(math_t x){
      return raft::myAbs(x);
    }, stream);
  MLCommon::Stats::colWeightedMean(out,error.data(), w, 1, len, stream);
}

template <typename math_t>
void _mean_square_error(cudaStream_t stream,int len, math_t* y, math_t* y_pred, math_t* w, math_t* out){
  rmm::device_uvector<math_t> error(len, stream);
  raft::linalg::subtract(error.data(), y, y_pred, len, stream);
  math_t power = 2.0;
  raft::linalg::unaryOp(error.data(), error.data(), len,
  [power] __device__(math_t x){
    return raft::myPow(x,power);
  }, stream);
  MLCommon::Stats::colWeightedMean(out,error.data(), w, 1, len, stream);
}

template <typename math_t>
void _root_mean_square_error(cudaStream_t stream, int len, math_t* y, math_t* y_pred,math_t* w, math_t* out){
  _mean_square_error(stream, len, y, y_pred, w, out);
  raft::linalg::unaryOp(
    out, out, 1, [] __device__(math_t in) { return raft::mySqrt(in); },
    stream);
}

template <typename math_t>
void _log_loss(cudaStream_t stream, int len, math_t* y, math_t* y_pred, math_t* w, math_t* out) {
  rmm::device_uvector<math_t> score(len, stream);
  MLCommon::Functions::logLoss(score.data(), y, y_pred, len, stream);
  MLCommon::Stats::colWeightedMean(out,score.data(), w, 1, len, stream);
}

template void _weighted_pearson<float>(cudaStream_t stream, int len, float* y , float* y_pred, float* w, float* out );
template void _weighted_spearman<float>(cudaStream_t stream, int len, float* y , float* y_pred, float* w, float* out );
template void _mean_absolute_error<float>(cudaStream_t stream, int len, float* y , float* y_pred, float* w, float* out );
template void _mean_square_error<float>(cudaStream_t stream, int len, float* y , float* y_pred, float* w, float* out );
template void _root_mean_square_error<float>(cudaStream_t stream, int len, float* y , float* y_pred, float* w, float* out );

}  // namespace genetic
}  // namespace cuml
