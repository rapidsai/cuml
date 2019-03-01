magma_int_t
magma_zgetri_outofplace_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magmaDoubleComplex **dinvA_array, magma_int_t lddia,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);
