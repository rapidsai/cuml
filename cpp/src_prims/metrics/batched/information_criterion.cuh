/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/**
 * @file information_criterion.cuh
 * @brief These information criteria are used to evaluate the quality of models
 *        by balancing the quality of the fit and the number of parameters.
 *
 * See:
 *  - AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion
 *  - AICc: https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
 *  - BIC: https://en.wikipedia.org/wiki/Bayesian_information_criterion
 */

#include <raft/linalg/unary_op.hpp>

#include <cmath>

namespace MLCommon {
namespace Metrics {

/// Supported types of information criteria
enum IC_Type { AIC, AICc, BIC };

namespace Batched {

/**
 * Compute the given type of information criterion
 *
 * @note: it is safe to do the computation in-place (i.e give same pointer
 *        as input and output)
 *
 * @param[out] d_ic             Information criterion to be returned for each
 *                              series (device)
 * @param[in]  d_loglikelihood  Log-likelihood for each series (device)
 * @param[in]  ic_type          Type of criterion to compute. See IC_Type
 * @param[in]  n_params         Number of parameters in the model
 * @param[in]  batch_size       Number of series in the batch
 * @param[in]  n_samples        Number of samples in each series
 * @param[in]  stream           CUDA stream
 */
template <typename ScalarT, typename IdxT>
void information_criterion(ScalarT* d_ic,
                           const ScalarT* d_loglikelihood,
                           IC_Type ic_type,
                           IdxT n_params,
                           IdxT batch_size,
                           IdxT n_samples,
                           cudaStream_t stream)
{
  ScalarT ic_base{};
  ScalarT N = static_cast<ScalarT>(n_params);
  ScalarT T = static_cast<ScalarT>(n_samples);
  switch (ic_type) {
    case AIC: ic_base = (ScalarT)2.0 * N; break;
    case AICc:
      ic_base = (ScalarT)2.0 * (N + (N * (N + (ScalarT)1.0)) / (T - N - (ScalarT)1.0));
      break;
    case BIC: ic_base = std::log(T) * N; break;
  }
  /* Compute information criterion from log-likelihood and base term */
  raft::linalg::unaryOp(
    d_ic,
    d_loglikelihood,
    batch_size,
    [=] __device__(ScalarT loglike) { return ic_base - (ScalarT)2.0 * loglike; },
    stream);
}

}  // namespace Batched
}  // namespace Metrics
}  // namespace MLCommon
