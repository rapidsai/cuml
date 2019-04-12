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


  const size_t ys_len = nobs;
  const size_t num_batches = h_Zb.size();

  ////////////////////////////////////////////////////////////
  // xfer from host to device

  //TODO: Far too many allocations for these. Definitely want to fix after getting this working

  double* d_ys;
  allocate(d_ys, num_batches*ys_len);
  for(int bi=0; bi<num_batches; bi++) {
    updateDevice(&d_ys[nobs * bi], h_ys_b[bi], nobs);
  }

  auto memory_pool = std::make_shared<BatchedMatrixMemoryPool>(num_batches);

  BatchedMatrix Zb(1, r, num_batches, memory_pool);
  BatchedMatrix Tb(r, r, num_batches, memory_pool);
  BatchedMatrix Rb(r, 1, num_batches, memory_pool);

  for(int bi=0; bi<num_batches; bi++) {
    updateDevice(Zb[bi], h_Zb[bi], r);

    updateDevice(Rb[bi], h_Rb[bi], r);

    updateDevice(Tb[bi], h_Tb[bi], r*r);
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
  allocate(d_vs, ys_len*num_batches);
  allocate(d_Fs, ys_len*num_batches);

  CUDA_CHECK(cudaPeekAtLastError());

  for(int it=0; it<ys_len; it++) {
    // std::cout << "it=" << it << " of " << ys_len << "\n";
    // 1.
    // vs[it] = ys[it] - alpha(0,0);
    // vs_eq_ys_m_alpha00(d_vs, it, d_ys_b, alpha);
    {
      auto counting = thrust::make_counting_iterator(0);
      double** d_alpha = alpha.data();
      thrust::for_each(counting, counting + num_batches,
                       [=]__device__(int bid) {
                         d_vs[bid + it*num_batches] = d_ys[it + bid*nobs] - d_alpha[bid][0];
                       });
    }

    // 2.
    // Fs[it] = P(0,0);
    // fs_it_P00(d_Fs, it, P);
    {
      double** d_P = P.data();
      auto counting = thrust::make_counting_iterator(0);
      thrust::for_each(counting, counting + num_batches,
                       [=]__device__(int bid) {
                         d_Fs[bid + it*num_batches] = d_P[bid][0];
                       });
    }

    // 3.
    // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
    BatchedMatrix TPZt = Tb * b_gemm(P, Zb, false, true);
    // BatchedMatrix K = _1_Fsit_TPZt(d_Fs, it, TPZt);
    BatchedMatrix K(r, 1, num_batches, memory_pool);
    {
      double** d_K = K.data();
      double** d_TPZt = TPZt.data();
      auto counting = thrust::make_counting_iterator(0);
      thrust::for_each(counting, counting + num_batches,
                       [=]__device__(int bid) {
                         for(int i=0; i<r; i++) {
                           d_K[bid][i] = 1.0/d_Fs[bid + it*num_batches] * d_TPZt[bid][i];
                         }
                       });
    }

    // 4.
    // alpha = T*alpha + K*vs[it];
    BatchedMatrix Kvs = Kvs_it(K, d_vs, it);
    alpha = Tb*alpha + Kvs;
    // std::cout << "alpha:" << alpha.shape().first << "," << alpha.shape().second << "\n";
    // 5.
    // MatrixT L = T - K*Z;
    BatchedMatrix L = Tb - K*Zb;

    // 6.
    // P = T * P * L.transpose() + R * R.transpose();
    P = Tb * b_gemm(P, L, false, true) + RRT;
    // std::cout << "P:" << P.shape().first << "," << P.shape().second << "\n";

  }

  // 7.
  // loglikelihood = sum(log(Fs[:]))
  double* loglikelihood = sumLogFs(d_Fs, num_batches, ys_len);

  // 8.
  // sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();

  double* sigma2;
  allocate(sigma2, num_batches);
  {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(counting, counting+num_batches,
                     [=]__device__(int bid) {
                       sigma2[bid] = 0.0;
                       double sigma2_sum = 0.0;
                       for (int it = 0; it < ys_len; it++) {
                         auto vsit = d_vs[bid + num_batches*it];
                         sigma2_sum += vsit*vsit / d_Fs[bid + num_batches*it];
                       }
                       sigma2[bid] = sigma2_sum / ys_len;
                     });
  }
  CUDA_CHECK(cudaPeekAtLastError());
  // 9.
  // loglike = -.5 * (loglikelihood + nobs * std::log(sigma2)) - nobs / 2. * (std::log(2 * M_PI) + 1);
  double* loglike;
  allocate(loglike, num_batches);
  {
    auto counting = thrust::make_counting_iterator(0);
    int nobs = ys_len;
    thrust::for_each(counting, counting+num_batches,
                     [=]__device__(int bid) {
                       loglike[bid] = -.5 * (loglikelihood[bid] + nobs * log(sigma2[bid]))
                         - nobs / 2. * (log(2 * M_PI) + 1);
                     });
  }
  CUDA_CHECK(cudaPeekAtLastError());
  ////////////////////////////////////////////////////////////
  // xfer results from GPU
  // need to fill:
  // vector<double*>& h_vs_b,   
  // vector<double*>& h_Fs_b,   
  // vector<double>& h_loglike_b
  // vector<double>& h_sigma2_b)
  //

  h_vs_b.resize(ys_len);
  h_Fs_b.resize(ys_len);
  h_loglike_b.resize(num_batches);
  h_sigma2_b.resize(num_batches);

  // vs, Fs
  vector<double> h_vs_raw(ys_len*num_batches);
  vector<double> h_Fs_raw(ys_len*num_batches);
  updateHost(h_vs_raw.data(), d_vs, ys_len*num_batches);
  updateHost(h_Fs_raw.data(), d_Fs, ys_len*num_batches);

  for(int it=0;it<ys_len;it++) {
    for (int bi = 0; bi < num_batches; bi++) {
      // note: vs, Fs computed in batch-major order. Return it in time-major order
      h_vs_b[bi][it] = h_vs_raw[bi + it * num_batches];
      h_Fs_b[bi][it] = h_Fs_raw[bi + it * num_batches];
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
  cudaFree(loglikelihood);
  CUDA_CHECK(cudaPeekAtLastError());

}
