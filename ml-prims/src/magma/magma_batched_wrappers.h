#include <magma_v2.h>

namespace MLCommon {
namespace LinAlg {
  template <typename T>
  void
  magmablas_gemm_batched(
          magma_trans_t transA, magma_trans_t transB,
          magma_int_t m, magma_int_t n, magma_int_t k,
          T alpha,
          T const * const * dA_array, magma_int_t ldda,
          T const * const * dB_array, magma_int_t lddb,
          T beta,
          T **dC_array, magma_int_t lddc,
          magma_int_t batchCount, magma_queue_t queue );

  template <>
  inline void
  magmablas_gemm_batched(
          magma_trans_t transA, magma_trans_t transB,
          magma_int_t m, magma_int_t n, magma_int_t k,
          float alpha,
          float const * const * dA_array, magma_int_t ldda,
          float const * const * dB_array, magma_int_t lddb,
          float beta,
          float **dC_array, magma_int_t lddc,
          magma_int_t batchCount, magma_queue_t queue )

  {
  return magmablas_sgemm_batched( transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue);
  }

  template <>
  inline void
  magmablas_gemm_batched(
          magma_trans_t transA, magma_trans_t transB,
          magma_int_t m, magma_int_t n, magma_int_t k,
          double alpha,
          double const * const * dA_array, magma_int_t ldda,
          double const * const * dB_array, magma_int_t lddb,
          double beta,
          double **dC_array, magma_int_t lddc,
          magma_int_t batchCount, magma_queue_t queue )

  {
  return magmablas_dgemm_batched( transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue);
  }

  template <typename T>
  magma_int_t
  magma_potrf_batched(
  magma_uplo_t uplo, magma_int_t n,
  T **dA_array, magma_int_t lda,
  magma_int_t *info_array,
  magma_int_t batchCount, magma_queue_t queue);

  template <>
  inline magma_int_t
  magma_potrf_batched(
  magma_uplo_t uplo, magma_int_t n,
  float **dA_array, magma_int_t lda,
  magma_int_t *info_array,
  magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_spotrf_batched( uplo, n, dA_array, lda, info_array, batchCount, queue);
  }

  template <>
  inline magma_int_t
  magma_potrf_batched(
  magma_uplo_t uplo, magma_int_t n,
  double **dA_array, magma_int_t lda,
  magma_int_t *info_array,
  magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_dpotrf_batched( uplo, n, dA_array, lda, info_array, batchCount, queue);
  }

  template <typename T>
  magma_int_t
  magma_getrf_batched(
      magma_int_t m, magma_int_t n,
      T **dA_array,
      magma_int_t lda,
      magma_int_t **ipiv_array,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue);

  template <>
  inline magma_int_t
  magma_getrf_batched(
      magma_int_t m, magma_int_t n,
      float **dA_array,
      magma_int_t lda,
      magma_int_t **ipiv_array,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_sgetrf_batched( m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue);
  }

  template <>
  inline magma_int_t
  magma_getrf_batched(
      magma_int_t m, magma_int_t n,
      double **dA_array,
      magma_int_t lda,
      magma_int_t **ipiv_array,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_dgetrf_batched( m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue);
  }

  template <typename T>
  magma_int_t
  magma_getri_outofplace_batched(
      magma_int_t n,
      T **dA_array, magma_int_t ldda,
      magma_int_t **dipiv_array,
      T **dinvA_array, magma_int_t lddia,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue);

  template <>
  inline magma_int_t
  magma_getri_outofplace_batched(
      magma_int_t n,
      float **dA_array, magma_int_t ldda,
      magma_int_t **dipiv_array,
      float **dinvA_array, magma_int_t lddia,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_sgetri_outofplace_batched( n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue);
  }

  template <>
  inline magma_int_t
  magma_getri_outofplace_batched(
      magma_int_t n,
      double **dA_array, magma_int_t ldda,
      magma_int_t **dipiv_array,
      double **dinvA_array, magma_int_t lddia,
      magma_int_t *info_array,
      magma_int_t batchCount, magma_queue_t queue)

  {
  return magma_dgetri_outofplace_batched( n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue);
  }



}
}
