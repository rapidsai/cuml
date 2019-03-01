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

