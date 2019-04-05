#include "kalman.h"
#include "batched_kalman.h"
#include <matrix/batched_matrix.h>
#include <utils.h>

using std::vector;

using MLCommon::Matrix::BatchedMatrix;
using MLCommon::Matrix::b_gemm;
using MLCommon::allocate;

__global__ void vs_eq_ys_m_alpha00_kernel(double* d_vs, int it,
                                          const double* ys_it,
                                          double** alpha, int r,
                                          int num_batches) {
  int batch_id = blockIdx.x*blockDim.x + threadIdx.x;
  if(batch_id < num_batches) {
    d_vs[it*num_batches + batch_id] = ys_it[batch_id] - alpha[batch_id][0];
  }
}

void vs_eq_ys_m_alpha00(double* d_vs,int it,const vector<double*>& ptr_ys_b,const BatchedMatrix& alpha) {
  const int num_batches = alpha.batches();
  const int block_size = 16;
  const int num_blocks = std::ceil((double)num_batches/(double)block_size);

  vs_eq_ys_m_alpha00_kernel<<<num_blocks, block_size>>>(d_vs, it, ptr_ys_b[it], alpha.data(), alpha.shape().first, num_batches);
  
}

__global__ void fs_it_P00_kernel(double* d_Fs, int it, double** P, int num_batches) {
  int batch_id = blockIdx.x*blockDim.x + threadIdx.x;
  if(batch_id < num_batches) {
    d_Fs[it*num_batches + batch_id] = P[batch_id][0];
  }
}

void fs_it_P00(double* d_Fs, int it, const BatchedMatrix& P) {

  const int block_size = 16;
  const int num_batches = P.batches();
  const int num_blocks = std::ceil((double)num_batches/(double)block_size);

  fs_it_P00_kernel<<<num_blocks, block_size>>>(d_Fs, it, P.data(), num_batches);

}

__global__ void _1_Fsit_TPZt_kernel(double* d_Fs, int it, double** TPZt,
                                    int N_TPZt, // size of matrix TPZt
                                    int num_batches,
                                    double** K // output
                                    ) {
  
  int batch_id = blockIdx.x;
  for(int i=0;i<N_TPZt/blockDim.x;i++) {
    int ij = threadIdx.x + i*blockDim.x;
    if(ij < N_TPZt) {
      K[batch_id][ij] = 1.0/d_Fs[batch_id + num_batches * it] * TPZt[batch_id][ij];
    }
  }
}

BatchedMatrix _1_Fsit_TPZt(double* d_Fs, int it, const BatchedMatrix& TPZt) {
  BatchedMatrix K(TPZt.shape().first, TPZt.shape().second, TPZt.batches());

  const int TPZt_size = TPZt.shape().first * TPZt.shape().second;
  const int block_size = (TPZt_size) % 128;
  
  const int num_batches = TPZt.batches();
  const int num_blocks = num_batches;

  // call kernel
  _1_Fsit_TPZt_kernel<<<num_blocks,block_size>>>(d_Fs, it, TPZt.data(), TPZt_size, num_batches, K.data());

  return K;
}

BatchedMatrix Kvs_it(const BatchedMatrix& K, double* d_vs, int it) {
  throw std::runtime_error("Not implemtend");
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
  const int block_size = 32;
  const int num_blocks = std::ceil((double)num_batches/(double)block_size);
  sumLogFs_kernel<<<num_blocks, block_size>>>(d_Fs, num_batches, nobs, d_sumLogFs);
  return d_sumLogFs;
}

void batched_kalman_filter(const vector<double*>& ptr_ys_b,
                           const vector<double*>& ptr_Zb,
                           const vector<double*>& ptr_Rb,
                           const vector<double*>& ptr_Tb,
                           int r,
                           vector<double*>& ptr_vs_b,
                           vector<double*>& ptr_Fs_b,
                           vector<double>& ptr_loglike_b,
                           vector<double>& ptr_sigma2_b) {


  const size_t num_batches = ptr_Zb.size();
  const size_t ys_len = ptr_ys_b.size();

  BatchedMatrix Zb(ptr_Zb, {1, r});
  BatchedMatrix Tb(ptr_Tb, {r, r});
  BatchedMatrix Rb(ptr_Rb, {r, 1});

  BatchedMatrix RRT = b_gemm(Rb, Rb, false, true);

  // MatrixT P = T * T.transpose() - T * Z.transpose() * Z * T.transpose() + R * R.transpose();
  BatchedMatrix P = b_gemm(Tb,Tb,true,false) - b_gemm(Tb,b_gemm(Zb,Tb,false,true),false,true) + RRT;

  // init alpha to zero
  BatchedMatrix alpha(r, 1, num_batches, true);

  // init vs, Fs
  // In batch-major format.
  double* d_vs;
  double* d_Fs;
  MLCommon::allocate(d_vs, ys_len*num_batches);
  MLCommon::allocate(d_Fs, ys_len*num_batches);

  for(int it=0; it<ys_len; it++) {

    // 1.
    // vs[it] = ys[it] - alpha(0,0);
    vs_eq_ys_m_alpha00(d_vs, it, ptr_ys_b, alpha);

    // 2.
    // Fs[it] = P(0,0);
    fs_it_P00(d_Fs, it, P);

    // 3.
    // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
    BatchedMatrix TPZt = Tb * b_gemm(P, Zb, false, true);
    BatchedMatrix K = _1_Fsit_TPZt(d_Fs, it, TPZt);

    // 4.
    // alpha = T*alpha + K*vs[it];
    BatchedMatrix Kvs = Kvs_it(K, d_vs, it);
    alpha = Tb*alpha + Kvs;

    // 5.
    // MatrixT L = T - K*Z;
    BatchedMatrix L = Tb - K*Zb;

    // 6.
    // P = T * P * L.transpose() + R * R.transpose();
    P = Tb * b_gemm(P, L, false, true) + RRT;

  }

  // 7.
  // loglikelihood = sum(log(Fs[:]))
  double* sumLogFs = sumLogFs(d_Fs, num_batches, ys_len);
  
  // 8.
  // sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();
  // 9.
  // loglike = -.5 * (loglikelihood + nobs * std::log(sigma2)) - nobs / 2. * (std::log(2 * M_PI) + 1);

  //xfer results from GPU

  // for(int i=0; i<num_batches; i++) {
  //   kalman_filter(ptr_ys_b[i], ys_len[i], ptr_Zb[i], ptr_Rb[i], ptr_Tb[i], r,
  //                 ptr_vs_b[0], ptr_Fs_b[0], &ptr_loglike_b[i], &ptr_sigma2_b[i]);
  // }

}
