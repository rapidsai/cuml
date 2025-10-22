/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/tsa/arima_common.h>

namespace raft {
class handle_t;
}

namespace ML {
namespace Datasets {

/**
 * Generates a dataset of time series by simulating an ARIMA process
 * of a given order.
 *
 * @param[in]  handle          cuML handle
 * @param[out] out             Generated time series
 * @param[in]  batch_size      Batch size
 * @param[in]  n_obs           Number of observations per series
 * @param[in]  order           ARIMA order
 * @param[in]  scale           Scale used to draw the starting values
 * @param[in]  noise_scale     Scale used to draw the residuals
 * @param[in]  intercept_scale Scale used to draw the intercept
 * @param[in]  seed            Seed for the random number generator
 * @{
 */
void make_arima(const raft::handle_t& handle,
                float* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                float scale           = 1.0f,
                float noise_scale     = 0.2f,
                float intercept_scale = 1.0f,
                uint64_t seed         = 0ULL);

void make_arima(const raft::handle_t& handle,
                double* out,
                int batch_size,
                int n_obs,
                ARIMAOrder order,
                double scale           = 1.0,
                double noise_scale     = 0.2,
                double intercept_scale = 1.0,
                uint64_t seed          = 0ULL);
/** @} */

}  // namespace Datasets
}  // namespace ML
