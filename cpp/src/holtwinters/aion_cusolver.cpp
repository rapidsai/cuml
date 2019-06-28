/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "aion_cusolver.hpp"

namespace aion {

thread_local cusolverDnHandle_t cusolver::m_handle = nullptr;

namespace {
cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, float *A, int lda,
                                           int *Lwork) {
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, double *A, int lda,
                                           int *Lwork) {
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t handle, int m, int n,
                                float *A, int lda, float *TAU, float *Workspace,
                                int Lwork, int *devInfo) {
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t handle, int m, int n,
                                double *A, int lda, double *TAU,
                                double *Workspace, int Lwork, int *devInfo) {
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, const float *A,
                                           int lda, const float *tau,
                                           int *lwork) {
  return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}
cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, const double *A,
                                           int lda, const double *tau,
                                           int *lwork) {
  return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                float *A, int lda, const float *tau,
                                float *work, int lwork, int *devInfo) {
  return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                double *A, int lda, const double *tau,
                                double *work, int lwork, int *devInfo) {
  return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

// TODO(ahmad): report mismatch between doc and API in cusolver
// cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
// int m, int n, int k, const float *A, int lda, int *lwork); // Doc
// cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
// int m, int n, int k, const float *A, int lda, const float *tau, int *lwork); // API

}  // namespace.

template <typename Dtype>
void cusolver::geqrf_bufferSize(int m, int n, Dtype *A, int lda, int *Lwork) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(cusolver_geqrf_bufferSize(handle, m, n, A, lda, Lwork));
}

template <typename Dtype>
void cusolver::geqrf(int m, int n, Dtype *A, int lda, Dtype *TAU,
                     Dtype *Workspace, int Lwork, int *devInfo) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_geqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
}

template <typename Dtype>
void cusolver::orgqr_bufferSize(int m, int n, int k, const Dtype *A, int lda,
                                const Dtype *tau, int *lwork) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_orgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}
template <typename Dtype>
void cusolver::orgqr(int m, int n, int k, Dtype *A, int lda, const Dtype *tau,
                     Dtype *work, int lwork, int *devInfo) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_orgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template void cusolver::geqrf_bufferSize(int m, int n, float *A, int lda,
                                         int *Lwork);
template void cusolver::geqrf_bufferSize(int m, int n, double *A, int lda,
                                         int *Lwork);
template void cusolver::geqrf(int m, int n, float *A, int lda, float *TAU,
                              float *Workspace, int Lwork, int *devInfo);
template void cusolver::geqrf(int m, int n, double *A, int lda, double *TAU,
                              double *Workspace, int Lwork, int *devInfo);

template void cusolver::orgqr_bufferSize(int m, int n, int k, const float *A,
                                         int lda, const float *tau, int *lwork);
template void cusolver::orgqr_bufferSize(int m, int n, int k, const double *A,
                                         int lda, const double *tau,
                                         int *lwork);
template void cusolver::orgqr(int m, int n, int k, float *A, int lda,
                              const float *tau, float *work, int lwork,
                              int *devInfo);
template void cusolver::orgqr(int m, int n, int k, double *A, int lda,
                              const double *tau, double *work, int lwork,
                              int *devInfo);

}  // namespace aion
