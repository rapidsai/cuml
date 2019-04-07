#pragma once

#include "magma/b_handles.h"

using namespace MLCommon;

namespace gmm {

template <typename T>
struct llhdHandle_t {

        T **dInvSigma_array, *dInvdet_array, **dX_batches, **dmu_batches, **dInvSigma_batches, **dDiff_batches, *dBil_batches;

        T* dInvSigma, *dDiff;
        size_t dDiff_size;

        int nCl, lddx, lddsigma_full, batchCount;

        T *bilinearWs, *determinantWs, *inverseWs;

        bilinearHandle_t<T> bilinearHandle;
        determinantHandle_t<T> determinantHandle;
        inverseHandle_t<T> inverseHandle;
};

}
