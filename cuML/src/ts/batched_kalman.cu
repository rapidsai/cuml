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
  dim3 grid(num_blocks,1,1);
  dim3 blk(block_size, 1, 1);

  vs_eq_ys_m_alpha00_kernel<<<blk, grid>>>(d_vs, it, ptr_ys_b[it], alpha.data(), alpha.shape().first, num_batches);
  
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
  dim3 grid(num_blocks,1,1);
  dim3 blk(block_size, 1, 1);

  fs_it_P00_kernel<<<blk, grid>>>(d_Fs, it, P.data(), num_batches);

}

BatchedMatrix _1_Fsit_TPZt(double* d_Fs, int it, const BatchedMatrix& TPZt) {
  BatchedMatrix K(TPZt.shape().first, TPZt.shape().second, TPZt.batches());

  const int block_size = 16;
  const int num_batches = TPZt.batches();
  const int num_blocks = std::ceil((double)num_batches/(double)block_size);
  dim3 grid(num_blocks,1,1);
  dim3 blk(block_size, 1, 1);

  // call kernel

  return K;
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

  // just use single kalman for now
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

  }

  // for(int i=0; i<num_batches; i++) {
  //   kalman_filter(ptr_ys_b[i], ys_len[i], ptr_Zb[i], ptr_Rb[i], ptr_Tb[i], r,
  //                 ptr_vs_b[0], ptr_Fs_b[0], &ptr_loglike_b[i], &ptr_sigma2_b[i]);
  // }

}
