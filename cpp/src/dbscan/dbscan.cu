/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "utils.h"
#include <cuML_api.h>
#include "runner.h"
#include "dbscan.hpp"
#include "dbscan.h"
#include <common/cumlHandle.hpp>

namespace ML {

using namespace Dbscan;

// @todo
// In the below 2 calls, the Index type has been hard-coded to `int64_t`
// We should pick the right Index type based on the input dimensions.
void dbscanFit(const cumlHandle& handle, float *input, int n_rows, int n_cols, float eps, int min_pts,
		       int *labels, size_t max_bytes_per_batch = 0, bool verbose) {
    dbscanFitImpl<float, int64_t>(handle.getImpl(), input, n_rows, n_cols, eps, min_pts, labels, max_bytes_per_batch, handle.getStream(), verbose);
}

void dbscanFit(const cumlHandle& handle, double *input, int n_rows, int n_cols, double eps, int min_pts,
		       int *labels, size_t max_bytes_per_batch = 0, bool verbose) {
    dbscanFitImpl<double, int64_t>(handle.getImpl(), input, n_rows, n_cols, eps, min_pts, labels, max_bytes_per_batch, handle.getStream(), verbose);
}

}; // end namespace ML


extern "C" cumlError_t cumlSpDbscanFit(cumlHandle_t handle, float *input, int n_rows, int n_cols, float eps, int min_pts,
               int *labels, size_t max_bytes_per_batch, bool verbose) {
    cumlError_t status;
    ML::cumlHandle *handle_ptr;
    std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
    if (status == CUML_SUCCESS) {
        try
        {
            dbscanFit(*handle_ptr, input, n_rows, n_cols, eps, min_pts, labels, max_bytes_per_batch, verbose);
        }
        //TODO: Implement this
        //catch (const MLCommon::Exception& e)
        //{
        //    //log e.what()?
        //    status =  e.getErrorCode();
        //}
        catch (...)
        {
            status = CUML_ERROR_UNKNOWN;
        }
    }
    return status;

}

extern "C" cumlError_t cumlDpDbscanFit(cumlHandle_t handle, double *input, int n_rows, int n_cols, double eps, int min_pts,
               int *labels, size_t max_bytes_per_batch, bool verbose) {
    cumlError_t status;
    ML::cumlHandle *handle_ptr;
    std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
    if (status == CUML_SUCCESS) {
        try
        {
            dbscanFit(*handle_ptr, input, n_rows, n_cols, eps, min_pts, labels, max_bytes_per_batch, verbose);
        }
        //TODO: Implement this
        //catch (const MLCommon::Exception& e)
        //{
        //    //log e.what()?
        //    status =  e.getErrorCode();
        //}
        catch (...)
        {
            status = CUML_ERROR_UNKNOWN;
        }
    }
    return status;
}
