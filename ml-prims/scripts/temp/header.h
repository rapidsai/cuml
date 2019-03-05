void
magmablas_zgemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex_ptr dA_array[], magma_int_t ldda,
        magmaDoubleComplex_ptr dx_array[], magma_int_t incx,
        magmaDoubleComplex beta,
        magmaDoubleComplex_ptr dy_array[], magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue);
