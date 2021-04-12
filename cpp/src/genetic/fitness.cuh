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
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/strided_reduction.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/unary_op.cuh>
#include <stats/weighted_mean.cuh>
#include <raft/stats/mean_center.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>

#include <raft/cudart_utils.h>
#include <functions/logisticReg.cuh>

namespace cuml {
namespace genetic {

template <typename math_t>
void _weighted_pearson(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){
  // Find Pearson's correlation coefficient

  cudaStream_t stream = h.get_stream();

  rmm::device_uvector<math_t> sample_corr(n_samples * n_progs, stream);  // Per sample correlation
  
  rmm::device_uvector<math_t> mu_Y(1, stream);                          // label mean
  rmm::device_uvector<math_t> mu_Y_pred(n_progs, stream);               // predicted label means

  rmm::device_uvector<math_t> Y_norm(n_samples, stream);                // normalized labels
  rmm::device_uvector<math_t> Y_pred_norm(n_samples * n_progs, stream); // normalized prediction labels
  
  rmm::device_uvector<math_t> Y_std(1, stream);                         // standard deviation of labels
  rmm::device_uvector<math_t> Y_pred_std(n_progs,stream);               // standard deviation of predicted labels


  // Find stats for Y
  MLCommon::Stats::colWeightedMean(mu_Y.data(),Y,W,1,n_samples,stream);
  raft::stats::meanCenter(Y_norm.data(), Y, mu_Y.data(), (math_t)1, n_samples, false, false, stream );
  raft::linalg::stridedReduction(Y_std.data(),Y_norm.data(),(math_t)1,n_samples,(math_t)0,stream,false,
                                [W]__device__(math_t v, int i){return v*v*W[i];},
                                raft::Sum<math_t>(),
                                [] __device__(math_t in){return raft::mySqrt(in);});

  math_t h_Y_std = Y_std.element(0,stream);

  // Find stats for Y_pred
  MLCommon::Stats::colWeightedMean(mu_Y_pred.data(),Y_pred,n_progs,n_samples,stream);
  raft::stats::meanCenter(Y_pred_norm.data(), Y_pred, mu_Y_pred.data(), n_progs, n_samples, false, false, stream);
  raft::linalg::stridedReduction(Y_pred_std.data(),Y_pred_norm.data(),n_progs,n_samples,(math_t)0,stream,false,
                                [W]__device__(math_t v, int i){return v*v*W[i];},
                                raft::Sum<math_t>(),
                                [] __device__(math_t in){return raft::mySqrt(in);});

  // Cross covariance
  raft::linalg::matrixVectorOp(sample_corr.data(), Y_pred_norm.data(), Y_norm.data(), W, n_progs, n_samples, false, false,
                              [] __device__(math_t y_pred, math_t y, math_t w){
                                return w * y_pred * y;
                              }, stream);
  
  // Find Correlation
  raft::linalg::stridedReduction(out,sample_corr.data(),n_progs,n_samples,(math_t)0,stream,false);
  raft::linalg::eltwiseDivideCheckZero(out,out,Y_pred_std.data(),n_progs,stream);
  raft::linalg::unaryOp(out,out,n_progs,[h_Y_std] __device__(math_t in){return in / h_Y_std;}, stream);
}

struct rank_functor{
  template<typename math_t>
  __host__ __device__ math_t operator()(math_t data){
    if(data == 0) return 0;
    else return 1;
  }
};

template <typename math_t>
void _weighted_spearman(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){
  cudaStream_t stream = h.get_stream();

  // Get ranks for Y
  thrust::device_vector<math_t> Ycopy(Y,Y+n_samples);
  thrust::device_vector<math_t> rank_idx(n_samples,0);
  thrust::device_vector<math_t> rank_diff(n_samples,0);  
  thrust::device_vector<math_t> Yrank(n_samples,0);
  
  thrust::sequence(thrust::cuda::par.on(stream),rank_idx.begin(),rank_idx.end(),0);
  thrust::sort_by_key(thrust::cuda::par.on(stream),Ycopy.begin(),Ycopy.end(),rank_idx.begin());
  thrust::adjacent_difference(thrust::cuda::par.on(stream),Ycopy.begin(),Ycopy.end(),rank_diff.begin());
  thrust::transform(thrust::cuda::par.on(stream),rank_diff.begin(),rank_diff.end(),rank_diff.begin(),rank_functor());
  rank_diff[0]=1;
  thrust::inclusive_scan(thrust::cuda::par.on(stream),rank_diff.begin(),rank_diff.end(),rank_diff.begin());
  thrust::copy(rank_diff.begin(),rank_diff.end(),thrust::make_permutation_iterator(Yrank.begin(),rank_idx.begin()));

  // Get ranks for Y_pred
  // TODO: Find a better way to batch this
  thrust::device_vector<math_t> Ypredcopy(Y_pred,Y_pred + n_samples*n_progs);
  thrust::device_vector<math_t> Ypredrank(n_samples*n_progs,0);
  thrust::device_ptr<math_t> Ypredptr = thrust::device_pointer_cast<math_t>(Ypredcopy.data());
  thrust::device_ptr<math_t> Ypredrankptr = thrust::device_pointer_cast<math_t>(Ypredrank.data());

  for(int i = 0; i < n_progs; ++i){
    thrust::sequence(thrust::cuda::par.on(stream),rank_idx.begin(),rank_idx.end(),0);
    thrust::sort_by_key(thrust::cuda::par.on(stream),Ypredptr + (i*n_samples),Ypredptr + ((i+1)*n_samples),rank_idx.begin());
    thrust::adjacent_difference(thrust::cuda::par.on(stream),Ypredptr + (i*n_samples),Ypredptr + ((i+1)*n_samples),rank_diff.begin());
    thrust::transform(thrust::cuda::par.on(stream),rank_diff.begin(),rank_diff.end(),rank_diff.begin(),rank_functor());
    rank_diff[0]=1;
    thrust::inclusive_scan(thrust::cuda::par.on(stream),rank_diff.begin(),rank_diff.end(),rank_diff.begin());
    thrust::copy(rank_diff.begin(),rank_diff.end(),thrust::make_permutation_iterator(Ypredrankptr + (i*n_samples),rank_idx.begin()));
  }
  
  // Compute pearson's coefficient
  _weighted_pearson(h,n_samples,n_progs,thrust::raw_pointer_cast(Yrank.data()),thrust::raw_pointer_cast(Ypredrank.data()),W,out);
} 

template <typename math_t>
void _mean_absolute_error(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){

  cudaStream_t stream = h.get_stream();
  rmm::device_uvector<math_t> error(n_samples*n_progs,stream);

  // Compute absolute differences
  raft::linalg::matrixVectorOp( error.data(),Y_pred,Y, n_progs, n_samples, false, false, 
                                [] __device__(math_t y_p, math_t y){
                                  return raft::myAbs(y_p - y);
                                },stream);

  // Average along rows
  MLCommon::Stats::colWeightedMean(out,error.data(), W, n_progs, n_samples, stream);
}

template <typename math_t>
void _mean_square_error(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){

  cudaStream_t stream = h.get_stream();
  rmm::device_uvector<math_t> error(n_samples*n_progs,stream);

  // Compute square differences
  raft::linalg::matrixVectorOp( error.data(),Y_pred,Y, n_progs, n_samples, false, false, 
                                [] __device__(math_t y_p, math_t y){
                                  return raft::myPow(y_p - y, (math_t)2);
                                },stream);
  
  // Add up row values per column
  MLCommon::Stats::colWeightedMean(out,error.data(),W,n_progs,n_samples,stream);
}

template <typename math_t>
void _root_mean_square_error(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){

  cudaStream_t stream = h.get_stream();
  
  // Find MSE
  _mean_square_error(h,n_samples,n_progs,Y,Y_pred,W,out);

  // Take sqrt on all entries
  raft::linalg::unaryOp(out,out,n_progs,[] __device__(math_t in){ return raft::mySqrt(in); },stream);
}

template <typename math_t>
void _log_loss(const raft::handle_t &h, const int n_samples, const int n_progs, const math_t* Y , const math_t* Y_pred, const math_t* W, math_t* out){

  cudaStream_t stream = h.get_stream();
  // Logistic error per sample
  rmm::device_uvector<math_t> error(n_samples*n_progs, stream);

  // Populate logistic error as matrix vector op
  raft::linalg::matrixVectorOp(error.data(),Y_pred,Y, n_progs, n_samples,false,false,
                               [] __device__(math_t y_p, math_t y){
                                 return -y * logf(y_p) -(1-y)*logf(1-y_p);
                               }, stream);

  // Take average along rows
  MLCommon::Stats::colWeightedMean(out,error.data(),W,n_progs,n_samples,stream);
}


template void _weighted_pearson<float>        (const raft::handle_t &h, const int n_samples, const int n_progs, const float* Y , const float* Y_pred, const float* W, float* out );
template void _weighted_spearman<float>       (const raft::handle_t &h, const int n_samples, const int n_progs, const float* Y , const float* Y_pred, const float* W, float* out );
template void _mean_absolute_error<float>     (const raft::handle_t &h, const int n_samples, const int n_progs, const float* Y , const float* Y_pred, const float* W, float* out );
template void _mean_square_error<float>       (const raft::handle_t &h, const int n_samples, const int n_progs, const float* Y , const float* Y_pred, const float* W, float* out );
template void _root_mean_square_error<float>  (const raft::handle_t &h, const int n_samples, const int n_progs, const float* Y , const float* Y_pred, const float* W, float* out );

}  // namespace genetic
}  // namespace cuml
