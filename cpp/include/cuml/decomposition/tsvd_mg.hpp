/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>
#include "tsvd.hpp"

#include <common/cumlHandle.hpp>

namespace ML {
namespace TSVD {
namespace opg {

/**
 * @brief performs MNMG fit operation for the tsvd
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param components: principal components of the input data
 * @output param singular_vals: singular values of the data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
void fit(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
         size_t n_parts, MLCommon::Matrix::floatData_t **input,
         float *components, float *singular_vals, paramsTSVD prms,
         bool verbose = false);

void fit(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
         size_t n_parts, MLCommon::Matrix::doubleData_t **input,
         double *components, double *singular_vals, paramsTSVD prms,
         bool verbose = false);

/**
 * @brief performs MNMG fit and transform operation for the tsvd.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param trans_input: transformed input data
 * @output param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
void fit_transform(cumlHandle &handle,
                   std::vector<MLCommon::Matrix::Data<float> *> &input_data,
                   MLCommon::Matrix::PartDescriptor &input_desc,
                   std::vector<MLCommon::Matrix::Data<float> *> &trans_data,
                   MLCommon::Matrix::PartDescriptor &trans_desc,
                   float *components, float *explained_var,
                   float *explained_var_ratio, float *singular_vals,
                   paramsTSVD prms, bool verbose);

void fit_transform(cumlHandle &handle,
                   std::vector<MLCommon::Matrix::Data<double> *> &input_data,
                   MLCommon::Matrix::PartDescriptor &input_desc,
                   std::vector<MLCommon::Matrix::Data<double> *> &trans_data,
                   MLCommon::Matrix::PartDescriptor &trans_desc,
                   double *components, double *explained_var,
                   double *explained_var_ratio, double *singular_vals,
                   paramsTSVD prms, bool verbose);

/**
 * @brief performs MNMG transform operation for the tsvd.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param trans_input: transformed input data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
void transform(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
               size_t n_parts, MLCommon::Matrix::Data<float> **input,
               float *components, MLCommon::Matrix::Data<float> **trans_input,
               paramsTSVD prms, bool verbose);

void transform(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
               size_t n_parts, MLCommon::Matrix::Data<double> **input,
               double *components, MLCommon::Matrix::Data<double> **trans_input,
               paramsTSVD prms, bool verbose);

/**
 * @brief performs MNMG inverse transform operation for the output.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param trans_input: transformed input data
 * @input param components: principal components of the input data
 * @output param input: input data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
void inverse_transform(cumlHandle &handle,
                       MLCommon::Matrix::RankSizePair **rank_sizes,
                       size_t n_parts,
                       MLCommon::Matrix::Data<float> **trans_input,
                       float *components, MLCommon::Matrix::Data<float> **input,
                       paramsTSVD prms, bool verbose);

void inverse_transform(cumlHandle &handle,
                       MLCommon::Matrix::RankSizePair **rank_sizes,
                       size_t n_parts,
                       MLCommon::Matrix::Data<double> **trans_input,
                       double *components,
                       MLCommon::Matrix::Data<double> **input, paramsTSVD prms,
                       bool verbose);

};  // end namespace opg
};  // namespace TSVD
};  // end namespace ML
