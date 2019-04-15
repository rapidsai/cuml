#include "kalman.h"
#include "batched_kalman.h"
#include <matrix/batched_matrix.h>
#include <utils.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <iostream>
#include <cstdio>

#include <fstream>
#include <unistd.h>

#include <nvToolsExt.h>

#include <cub/cub.cuh>

// #include <thrust/lo

using std::vector;

using MLCommon::Matrix::BatchedMatrix;
using MLCommon::Matrix::BatchedMatrixMemoryPool;
using MLCommon::Matrix::b_gemm;
using MLCommon::allocate;
using MLCommon::updateDevice;
using MLCommon::updateHost;

////////////////////////////////////////////////////////////
#include <iostream>

void process_mem_usage(double& vm_usage, double& resident_set)
{
  vm_usage     = 0.0;
  resident_set = 0.0;

  // the two fields we want
  unsigned long vsize;
  long rss;
  {
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> vsize >> rss;
  }

  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
}
////////////////////////////////////////////////////////////


// __global__ void vs_eq_ys_m_alpha00_kernel(double* d_vs, int it,
//                                           const double* ys_it,
//                                           double** alpha, int r,
//                                           int num_batches) {
//   int batch_id = blockIdx.x*blockDim.x + threadIdx.x;
//   if(batch_id < num_batches) {
//     d_vs[it*num_batches + batch_id] = ys_it[batch_id] - alpha[batch_id][0];
//   }
// }

// void vs_eq_ys_m_alpha00(double* d_vs,int it,const vector<double*>& ptr_ys_b,const BatchedMatrix& alpha) {
//   const int num_batches = alpha.batches();
//   const int block_size = 16;
//   const int num_blocks = std::ceil((double)num_batches/(double)block_size);

//   vs_eq_ys_m_alpha00_kernel<<<num_blocks, block_size>>>(d_vs, it, ptr_ys_b[it],
//                                                         alpha.data(), alpha.shape().first, num_batches);
//   CUDA_CHECK(cudaPeekAtLastError());
  
// }

// __global__ void fs_it_P00_kernel(double* d_Fs, int it, double** P, int num_batches) {
//   int batch_id = blockIdx.x*blockDim.x + threadIdx.x;
//   if(batch_id < num_batches) {
//     d_Fs[it*num_batches + batch_id] = P[batch_id][0];
//   }
// }

// void fs_it_P00(double* d_Fs, int it, const BatchedMatrix& P) {

//   const int block_size = 16;
//   const int num_batches = P.batches();
//   const int num_blocks = std::ceil((double)num_batches/(double)block_size);

//   fs_it_P00_kernel<<<num_blocks, block_size>>>(d_Fs, it, P.data(), num_batches);
//   CUDA_CHECK(cudaPeekAtLastError());

// }

// __global__ void _1_Fsit_TPZt_kernel(double* d_Fs, int it, double** TPZt,
//                                     int N_TPZt, // size of matrix TPZt
//                                     int num_batches,
//                                     double** K // output
//                                     ) {
  
//   int batch_id = blockIdx.x;
//   for(int i=0;i<N_TPZt/blockDim.x;i++) {
//     int ij = threadIdx.x + i*blockDim.x;
//     if(ij < N_TPZt) {
//       K[batch_id][ij] = 1.0/d_Fs[batch_id + num_batches * it] * TPZt[batch_id][ij];
//     }
//   }
// }

// BatchedMatrix _1_Fsit_TPZt(double* d_Fs, int it, const BatchedMatrix& TPZt) {
//   BatchedMatrix K(TPZt.shape().first, TPZt.shape().second, TPZt.batches());

//   const int TPZt_size = TPZt.shape().first * TPZt.shape().second;
//   const int block_size = (TPZt_size) % 128;
  
//   const int num_batches = TPZt.batches();
//   const int num_blocks = num_batches;

//   // call kernel
//   _1_Fsit_TPZt_kernel<<<num_blocks,block_size>>>(d_Fs, it, TPZt.data(), TPZt_size, num_batches, K.data());
//   CUDA_CHECK(cudaPeekAtLastError());

//   return K;
// }

BatchedMatrix Kvs_it(const BatchedMatrix& K, double* d_vs, int it) {
  BatchedMatrix Kvs(K.shape().first, K.shape().second, K.batches(), K.pool());
  auto num_batches = K.batches();
  auto counting = thrust::make_counting_iterator(0);
  double** d_K = K.data();
  double** d_Kvs = Kvs.data();
  int m = K.shape().first;
  int n = K.shape().second;
  thrust::for_each(counting, counting + num_batches,
                   [=]__device__(int bid) {
                     double vs = d_vs[bid + it*num_batches];
                     for(int ij=0; ij<m*n; ij++) {
                       d_Kvs[bid][ij] = d_K[bid][ij]*vs;
                     }
                   });
  return Kvs;
}

__global__ void sumLogFs_kernel(double* d_Fs, int num_batches, int nobs, double* d_sumLogFs) {
  double sum = 0.0;
  int bid = threadIdx.x + blockIdx.x*blockDim.x;
  if(bid < num_batches) {
    for (int it = 0; it < nobs; it++) {
      sum += log(d_Fs[bid]);
    }
    d_sumLogFs[bid] = sum;
  }
}

double* sumLogFs(double* d_Fs, const int num_batches, const int nobs) {

  double* d_sumLogFs;
  allocate(d_sumLogFs, num_batches);
  // compute sum(log(Fs[0:nobs]))
  // const int block_size = 32;
  // const int num_blocks = std::ceil((double)num_batches/(double)block_size);
  // sumLogFs_kernel<<<num_blocks, block_size>>>(d_Fs, num_batches, nobs, d_sumLogFs);
  // CUDA_CHECK(cudaPeekAtLastError());
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(counting, counting + num_batches,
                   [=]__device__(int bid) {
                     double sum = 0.0;
                     for (int it = 0; it < nobs; it++) {
                       sum += log(d_Fs[bid + it*num_batches]);
                     }
                     d_sumLogFs[bid] = sum;
                   });
  CUDA_CHECK(cudaPeekAtLastError());
  return d_sumLogFs;
}


void batched_kalman_filter_cpu(const vector<double*>& h_ys_b, // { vector size batches, each item size nobs }
                               int nobs,
                               const vector<double*>& h_Zb, // { vector size batches, each item size Zb }
                               const vector<double*>& h_Rb, // { vector size batches, each item size Rb }
                               const vector<double*>& h_Tb, // { vector size batches, each item size Tb }
                               int r,
                               vector<double*>& h_vs_b, // { vector size batches, each item size nobs }
                               vector<double*>& h_Fs_b, // { vector size batches, each item size nobs }
                               vector<double>& h_loglike_b,
                               vector<double>& h_sigma2_b) {

  nvtxNameOsThread(0, "MAIN");
  nvtxRangePush(__FUNCTION__);
  nvtxMark("batched_kalman_cpu");

  const size_t num_batches = h_Zb.size();
  for(int bi=0; bi<num_batches; bi++) {
    kalman_filter(h_ys_b[bi], nobs,
                  h_Zb[bi], h_Rb[bi], h_Tb[bi],
                  r,
                  h_vs_b[bi], h_Fs_b[bi],
                  &h_loglike_b[bi], &h_sigma2_b[bi]
                  );
  }
  nvtxRangePop();
}

__device__ void Mv(double* A, double* v, int r, int tid, double* out) {
  if(tid < r) {
    out[tid] = 0.0;
    for(int i=0; i<r; i++) {
      out[tid] += A[tid + r*i] * v[i];
    }
  }
}

__device__ void MM(double *A, double *B, int r, int tid, double *out) {

  out[tid] = 0.0;
  for(int i=0; i<r; i++) {
    
    // access pattern should be:
    // out[0] += A[0 + r*i] * B[i + 0*r];
    // out[1] += A[1 + r*i] * B[i + 0*r];
    // out[2] += A[0 + r*i] * B[i + 1*r];
    // out[3] += A[1 + r*i] * B[i + 1*r];
    
    out[tid] += A[tid%r + r*i]*B[i + (tid/r % r) *r];
  }

}

extern __shared__ double s_array[]; // size = r*r x 5 + r x 3
__global__ void batched_kalman_loop_kernel(double* ys, int nobs,
                                           double** T, // \in R^(r x r)
                                           double** Z, // \in R^(1 x r)
                                           double** RRT, // \in R^(r x r)
                                           double** P, // \in R^(r x r)
                                           double** alpha, // \in R^(r x 1)
                                           int r,
                                           int num_batches,
                                           double* vs,
                                           double* Fs,
                                           double* sum_logFs
                                           ) {

  // kalman matrices and temporary storage
  int r2 = r*r;
  double* s_RRT = &s_array[0]; // rxr
  double* s_T = &s_array[r2]; // rxr
  double* s_Z = &s_array[2*r2]; // r
  double* s_P = &s_array[2*r2+r]; // rxr
  double* s_alpha = &s_array[3*r2+r]; // r
  double* s_K = &s_array[3*r2+2*r]; // r
  double* tmpA = &s_array[3*r2+3*r]; // rxr
  double* tmpB = &s_array[4*r2+3*r]; // rxr

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // preload kalman matrices from GM.
  s_RRT[tid] = RRT[bid][tid];
  s_T[tid] = T[bid][tid];
  s_P[tid] = P[bid][tid];
  if(tid < r) {
    s_Z[tid] = Z[bid][tid];
    s_alpha[tid]= alpha[bid][tid];
  }
  __syncthreads();

  double bid_sum_logFs = 0.0;

  for(int it=0; it<nobs; it++) {

    // 1. & 2.
    // vs[it] = ys[it] - alpha(0,0);
    // Fs[it] = P(0,0);
    if(tid==0) {
      vs[it + bid*nobs] = ys[it + bid*nobs] - s_alpha[0];
      Fs[it + bid*nobs] = s_P[0];
      bid_sum_logFs += log(s_P[0]);
    }
    __syncthreads();
  
    // 3.
    // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
    // tmpA = P*Z.T
    Mv(s_P, s_Z, r, tid, tmpA);
    // tmpB = T*tmpA
    Mv(s_T, tmpA, r, tid, tmpB);
    // tmpB = 1/Fs[it] * tmpB
    if(tid < r) {
      s_K[tid] = 1/Fs[it + bid*nobs] * tmpB[tid];
    }
    __syncthreads();
  

    // 4.
    // alpha = T*alpha + K*vs[it];
    if (tid < r) {
      Mv(s_T, s_alpha, r, tid, tmpA);
      s_alpha[tid] = tmpA[tid] + s_K[tid] * vs[it + bid * nobs];
    }
    __syncthreads();

    // 5.
    // MatrixT L = T - K*Z;
    // tmpA = KZ
    // tmpA[0] = K[0]*Z[0]
    // tmpA[1] = K[1]*Z[0]
    // tmpA[2] = K[0]*Z[1]
    // tmpA[3] = K[1]*Z[1]
    // pytest [i//3 % 3 for i in range(9)] -> 0 1 2 0 1 2 0 1 2
    // pytest [i//3 % 3 for i in range(9)] -> 0 0 0 1 1 1 2 2 2

    tmpA[tid] = s_K[tid % r] * s_Z[tid / r % r];

    __syncthreads();
    // tmpA = T-tmpA
    tmpA[tid] = s_T[tid] - tmpA[tid];
    __syncthreads();
    // L = tmpA

    // 6.
    // tmpB = tmpA.transpose()
    tmpB[tid] = tmpA[tid * r + tid / r % r];
    // L.T = tmpB
    __syncthreads();

    // P = T * P * L.transpose() + R * R.transpose();
    // tmpA = P*L.T
    MM(s_P, tmpB, r, tid, tmpA);
    __syncthreads();
    // tmpB = T*tmpA;
    MM(s_T, tmpA, r, tid, tmpB);

    // P = tmpB + RRT
    s_P[tid] = tmpB[tid] + s_RRT[tid];
    __syncthreads();
  }
  if(tid == 0) {
    sum_logFs[bid] = bid_sum_logFs;
  }
}

void batched_kalman_loop(double* ys, int nobs,
                         const BatchedMatrix& T,
                         const BatchedMatrix& Z,
                         const BatchedMatrix& RRT,
                         const BatchedMatrix& P0,
                         const BatchedMatrix& alpha,
                         int r,
                         double* vs,
                         double* Fs,
                         double* sum_logFs
                         ) {

  const int num_batches = T.batches();
  const int num_blocks = num_batches;
  const int num_threads = r*r;
  const size_t bytes_shared_memory = (5*r*r + 3*r) * sizeof(double);
  
  batched_kalman_loop_kernel<<<num_blocks, num_threads, bytes_shared_memory>>>(ys, nobs,
                                                                               T.data(), Z.data(),
                                                                               RRT.data(), P0.data(),
                                                                               alpha.data(),
                                                                               r,
                                                                               num_batches,
                                                                               vs, Fs,
                                                                               sum_logFs
                                                                               );

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

}

__global__ void batched_kalman_loglike_kernel(double *d_vs, double *d_Fs, double *d_sumLogFs,
                               int nobs, int num_batches, double *sigma2,
                               double *loglike) {

  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads = blockDim.x;
  double bid_sigma2 = 0.0;
  for(int it=0; it<nobs; it+=num_threads) {
    // vs and Fs are in time-major order
    int idx = (it + tid) + bid * nobs;
    double d_vs2_Fs = 0.0;
    if (idx < nobs*num_batches) {
      d_vs2_Fs = d_vs[idx] * d_vs[idx] / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if(tid == 0) {
    bid_sigma2 /= nobs;
    sigma2[bid] = bid_sigma2;
    loglike[bid] = -.5 * (d_sumLogFs[bid] + nobs * log(bid_sigma2)) - nobs / 2. * (log(2 * M_PI) + 1);
  }
}

void batched_kalman_loglike(double* d_vs, double* d_Fs, double* d_sumLogFs, int nobs, int num_batches,
                    double* sigma2, double* loglike) {

  // BlockReduce uses 128 threads, so here also use 128 threads.
  const int num_threads = 128;
  batched_kalman_loglike_kernel<<<num_batches, num_threads>>>(d_vs, d_Fs, d_sumLogFs, nobs, num_batches,
                                                              sigma2, loglike);
  CUDA_CHECK(cudaDeviceSynchronize());

}

void batched_kalman_filter(const vector<double*>& h_ys_b, // { vector size batches, each item size nobs }
                           int nobs,
                           const vector<double*>& h_Zb, // { vector size batches, each item size Zb }
                           const vector<double*>& h_Rb, // { vector size batches, each item size Rb }
                           const vector<double*>& h_Tb, // { vector size batches, each item size Tb }
                           int r,
                           vector<double*>& h_vs_b,
                           vector<double*>& h_Fs_b,
                           vector<double>& h_loglike_b,
                           vector<double>& h_sigma2_b) {

  nvtxNameOsThread(0, "MAIN");
  nvtxRangePush(__FUNCTION__);
  const size_t ys_len = nobs;
  const size_t num_batches = h_Zb.size();

  ////////////////////////////////////////////////////////////
  // xfer from host to device

  //TODO: Far too many allocations for these. Definitely want to fix after getting this working

  double* d_ys;
  allocate(d_ys, num_batches*ys_len);
  {
    std::vector<double> h_ys(num_batches*ys_len);
    for(int bi=0; bi<num_batches; bi++) {
      for (int it = 0; it < nobs; it++) {
        h_ys[it + bi * nobs] = h_ys_b[bi][it];
      }
    }
    updateDevice(d_ys, h_ys.data(), h_ys.size());
  }
  auto memory_pool = std::make_shared<BatchedMatrixMemoryPool>(num_batches);

  BatchedMatrix Zb(1, r, num_batches, memory_pool);
  BatchedMatrix Tb(r, r, num_batches, memory_pool);
  BatchedMatrix Rb(r, 1, num_batches, memory_pool);
  {
    //Tb
    std::vector<double> matrix_copy(r*r*num_batches);
    for(int bi=0;bi<num_batches;bi++) {
      for(int i=0;i<r*r;i++) {
        matrix_copy[i + bi*r*r] = h_Tb[bi][i];
      }
    }
    updateDevice(Tb[0],matrix_copy.data(),r*r*num_batches);

    //Zb
    for(int bi=0;bi<num_batches;bi++) {
      for(int i=0;i<r;i++) {
        matrix_copy[i + bi*r] = h_Zb[bi][i];
      }
    }
    updateDevice(Zb[0],matrix_copy.data(),r*num_batches);

    // Rb
    for(int bi=0;bi<num_batches;bi++) {
      for(int i=0;i<r;i++) {
        matrix_copy[i + bi*r] = h_Rb[bi][i];
      }
    }
    updateDevice(Rb[0],matrix_copy.data(),r*num_batches);
  }
  
  CUDA_CHECK(cudaPeekAtLastError());

  ////////////////////////////////////////////////////////////
  // Computation
  
  BatchedMatrix RRT = b_gemm(Rb, Rb, false, true);

  // MatrixT P = T * T.transpose() - T * Z.transpose() * Z * T.transpose() + R * R.transpose();
  BatchedMatrix P = b_gemm(Tb,Tb,false,true) - Tb * b_gemm(Zb,b_gemm(Zb,Tb,false,true),true,false) + RRT;

  // init alpha to zero
  BatchedMatrix alpha(r, 1, num_batches, memory_pool, true);

  // init vs, Fs
  // In batch-major format.
  double* d_vs;
  double* d_Fs;
  double* d_sumlogFs;
  allocate(d_vs, ys_len*num_batches);
  allocate(d_Fs, ys_len*num_batches);
  allocate(d_sumlogFs, num_batches);

  CUDA_CHECK(cudaPeekAtLastError());

  
  // Reference implementation
  // For it = 1:nobs
  //  // 1.
  //   vs[it] = ys[it] - alpha(0,0);
  //  // 2.
  //   Fs[it] = P(0,0);

  //   if(Fs[it] < 0) {
  //     std::cout << "P=" << P << "\n";
  //     throw std::runtime_error("ERROR: F < 0");
  //   }
  //   3.
  //   MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
  //   4.
  //   alpha = T*alpha + K*vs[it];
  //   5.
  //   MatrixT L = T - K*Z;
  //   6.
  //   P = T * P * L.transpose() + R * R.transpose();
  //   loglikelihood += std::log(Fs[it]);
  // }

  batched_kalman_loop(d_ys, nobs, Tb, Zb, RRT, P, alpha, r, d_vs, d_Fs, d_sumlogFs);

  // Finalize loglikelihood
  // 7. & 8.
  // double sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();
  // double loglike = -.5 * (loglikelihood + nobs * std::log(sigma2));
  // loglike -= nobs / 2. * (std::log(2 * M_PI) + 1);
  double* sigma2;
  allocate(sigma2, num_batches);
  double* loglike;
  allocate(loglike, num_batches);
  batched_kalman_loglike(d_vs, d_Fs, d_sumlogFs, nobs, num_batches, sigma2, loglike);
  
  ////////////////////////////////////////////////////////////
  // xfer results from GPU
  // need to fill:
  // vector<double*>& h_vs_b,   
  // vector<double*>& h_Fs_b,   
  // vector<double>& h_loglike_b
  // vector<double>& h_sigma2_b)

  h_vs_b.resize(ys_len);
  h_Fs_b.resize(ys_len);
  h_loglike_b.resize(num_batches);
  h_sigma2_b.resize(num_batches);

  // vs, Fs
  vector<double> h_vs_raw(ys_len*num_batches);
  vector<double> h_Fs_raw(ys_len*num_batches);
  updateHost(h_vs_raw.data(), d_vs, ys_len*num_batches);
  updateHost(h_Fs_raw.data(), d_Fs, ys_len*num_batches);

  for (int bi = 0; bi < num_batches; bi++) {
    for(int it=0;it<ys_len;it++) {
      h_vs_b[bi][it] = h_vs_raw[it + bi * nobs];
      h_Fs_b[bi][it] = h_Fs_raw[it + bi * nobs];
    }
  }

  updateHost(h_loglike_b.data(), loglike, num_batches);
  updateHost(h_sigma2_b.data(), sigma2, num_batches);
  CUDA_CHECK(cudaPeekAtLastError());
  ////////////////////////////////////////////////////////////
  // free memory
  cudaFree(d_ys);
  cudaFree(d_vs);
  cudaFree(d_Fs);
  cudaFree(sigma2);
  cudaFree(loglike);
  CUDA_CHECK(cudaPeekAtLastError());
  nvtxRangePop();
}
