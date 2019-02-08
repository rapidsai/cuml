#pragma once

namespace gmm {
template <typename T>
struct Variables {
        // these device pointers are NOT owned by this class
        T *z, *x;
        T *mus, *sigmas, *rhos; // predict_P

        bool initialized=false;
        cublasHandle_t handle;

        int n_classes;
        int dim_x;

        // all the workspace related pointers
        int *info;
};


}; // end namespace gmm
