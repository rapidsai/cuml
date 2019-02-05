import numpy as np

cdef extern from "kmeans/kmeans_c.h" namespace "ML":

    cdef void make_ptr_kmeans(
        int dopredict,
        int verbose,
        int seed,
        int gpu_id,
        int n_gpu,
        size_t mTrain,
        size_t n,
        const char ord,
        int k,
        int k_max,
        int max_iterations,
        int init_from_data,
        float threshold,
        const float *srcdata,
        const float *centroids,
        float *pred_centroids,
        int *pred_labels
    )

    cdef void make_ptr_kmeans(
        int dopredict,
        int verbose,
        int seed,
        int gpu_id,
        int n_gpu,
        size_t mTrain,
        size_t n,
        const char ord,
        int k,
        int k_max,
        int max_iterations,
        int init_from_data,
        double threshold,
        const double *srcdata,
        const double *centroids,
        double *pred_centroids,
        int *pred_labels
    )


    cdef void kmeans_transform(int verbose,
                             int gpu_id, int n_gpu,
                             size_t m, size_t n, const char ord, int k,
                             const float *src_data, const float *centroids,
                             float *preds)

    cdef void kmeans_transform(int verbose,
                              int gpu_id, int n_gpu,
                              size_t m, size_t n, const char ord, int k,
                              const double *src_data, const double *centroids,
                              double *preds)

