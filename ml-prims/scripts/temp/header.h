void
magmablas_zgemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
        magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex_ptr dC, magma_int_t lddc,
        magma_queue_t queue );
