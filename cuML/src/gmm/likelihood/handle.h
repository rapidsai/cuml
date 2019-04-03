#pragma once

#include "magma/b_handles.h"

using namespace MLCommon;

namespace gmm {

template <typename T>
struct llhdHandle_t {
        T **dInvSigma_array=NULL, *dInvdet_array=NULL,
        **dX_batches=NULL, **dmu_batches=NULL,
        **dInvSigma_batches=NULL, **dDiff_batches=NULL,
        *dBil_batches=NULL;

        bilinearHandle_t<T> bilinearHandle;
        determinantHandle_t<T> determinantHandle;
        inverseHandle_t<T> inverseHandle;

};

}
