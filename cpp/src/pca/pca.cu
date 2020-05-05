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

#include <cuml/decomposition/pca.hpp>
#include "pca.cuh"

namespace ML {

using namespace MLCommon;

void pcaFit(cumlHandle &handle, float *input, float *components,
            float *explained_var, float *explained_var_ratio,
            float *singular_vals, float *mu, float *noise_vars,
            const paramsPCA &prms) {
  pcaFit(handle.getImpl(), input, components, explained_var,
         explained_var_ratio, singular_vals, mu, noise_vars, prms,
         handle.getStream());
}

void pcaFit(cumlHandle &handle, double *input, double *components,
            double *explained_var, double *explained_var_ratio,
            double *singular_vals, double *mu, double *noise_vars,
            const paramsPCA &prms) {
  pcaFit(handle.getImpl(), input, components, explained_var,
         explained_var_ratio, singular_vals, mu, noise_vars, prms,
         handle.getStream());
}

void pcaFitTransform(cumlHandle &handle, float *input, float *trans_input,
                     float *components, float *explained_var,
                     float *explained_var_ratio, float *singular_vals,
                     float *mu, float *noise_vars, const paramsPCA &prms) {
  pcaFitTransform(handle.getImpl(), input, trans_input, components,
                  explained_var, explained_var_ratio, singular_vals, mu,
                  noise_vars, prms, handle.getStream());
}

void pcaFitTransform(cumlHandle &handle, double *input, double *trans_input,
                     double *components, double *explained_var,
                     double *explained_var_ratio, double *singular_vals,
                     double *mu, double *noise_vars, const paramsPCA &prms) {
  pcaFitTransform(handle.getImpl(), input, trans_input, components,
                  explained_var, explained_var_ratio, singular_vals, mu,
                  noise_vars, prms, handle.getStream());
}

void pcaInverseTransform(cumlHandle &handle, float *trans_input,
                         float *components, float *singular_vals, float *mu,
                         float *input, const paramsPCA &prms) {
  pcaInverseTransform(handle.getImpl(), trans_input, components, singular_vals,
                      mu, input, prms, handle.getStream());
}

void pcaInverseTransform(cumlHandle &handle, double *trans_input,
                         double *components, double *singular_vals, double *mu,
                         double *input, const paramsPCA &prms) {
  pcaInverseTransform(handle.getImpl(), trans_input, components, singular_vals,
                      mu, input, prms, handle.getStream());
}

void pcaTransform(cumlHandle &handle, float *input, float *components,
                  float *trans_input, float *singular_vals, float *mu,
                  const paramsPCA &prms) {
  pcaTransform(handle.getImpl(), input, components, trans_input, singular_vals,
               mu, prms, handle.getStream());
}

void pcaTransform(cumlHandle &handle, double *input, double *components,
                  double *trans_input, double *singular_vals, double *mu,
                  const paramsPCA &prms) {
  pcaTransform(handle.getImpl(), input, components, trans_input, singular_vals,
               mu, prms, handle.getStream());
}

};  // end namespace ML
