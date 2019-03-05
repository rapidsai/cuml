template <typename T>
void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        T alpha,
        T** dA_array, magma_int_t ldda,
        T** dx_array, magma_int_t incx,
        T beta,
        T** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue);

template <>
inline void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        float alpha,
        float** dA_array, magma_int_t ldda,
        float** dx_array, magma_int_t incx,
        float beta,
        float** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magmablas_sgemv_batched( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
}

template <>
inline void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        double alpha,
        double** dA_array, magma_int_t ldda,
        double** dx_array, magma_int_t incx,
        double beta,
        double** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magmablas_dgemv_batched( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
}
