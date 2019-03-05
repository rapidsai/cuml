template <typename T>
void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        T alpha,
        T_const_ptr dA, magma_int_t ldda,
        T_const_ptr dB, magma_int_t lddb,
        T beta,
        T_ptr dC, magma_int_t lddc,
        magma_queue_t queue );

template <>
inline void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        float alpha,
        float_const_ptr dA, magma_int_t ldda,
        float_const_ptr dB, magma_int_t lddb,
        float beta,
        float_ptr dC, magma_int_t lddc,
        magma_queue_t queue )

{
        return magmablas_sgemm( transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
}

template <>
inline void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        double alpha,
        double_const_ptr dA, magma_int_t ldda,
        double_const_ptr dB, magma_int_t lddb,
        double beta,
        double_ptr dC, magma_int_t lddc,
        magma_queue_t queue )

{
        return magmablas_dgemm( transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
}
