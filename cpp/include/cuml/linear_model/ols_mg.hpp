/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
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

#include <cuml/linear_model/glm.hpp>
#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>

#include <common/cumlHandle.hpp>

namespace ML {
namespace OLS {
namespace opg {

/**
 * @brief performs MNMG fit operation for the ridge regression
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param n_rows: number of rows of the input data
 * @input param n_cols: number of cols of the input data
 * @input param labels: labels data
 * @output param coef: learned regression coefficients
 * @output param intercept: intercept value
 * @input param fit_intercept: fit intercept or not
 * @input param normalize: normalize the data or not
 * @input param algo: which algorithm is used for OLS. 0 is for SVD, 1 is for eig.
 * @input param verbose
 */
void fit(cumlHandle &handle,
         std::vector<MLCommon::Matrix::Data<float> *> &input_data,
         MLCommon::Matrix::PartDescriptor &input_desc,
         std::vector<MLCommon::Matrix::Data<float> *> &labels, float *coef,
         float *intercept, bool fit_intercept, bool normalize, int algo,
         bool verbose);

void fit(cumlHandle &handle,
         std::vector<MLCommon::Matrix::Data<double> *> &input_data,
         MLCommon::Matrix::PartDescriptor &input_desc,
         std::vector<MLCommon::Matrix::Data<double> *> &labels, double *coef,
         double *intercept, bool fit_intercept, bool normalize, int algo,
         bool verbose);

/**
 * @brief performs MNMG prediction for OLS
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param n_rows: number of rows of input data
 * @input param n_cols: number of cols of input data
 * @input param coef: OLS coefficients
 * @output param preds: predictions
 * @input param verbose
 */
void predict(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
             size_t n_parts, MLCommon::Matrix::Data<float> **input,
             size_t n_rows, size_t n_cols, float *coef, float intercept,
             MLCommon::Matrix::Data<float> **preds, bool verbose);

void predict(cumlHandle &handle, MLCommon::Matrix::RankSizePair **rank_sizes,
             size_t n_parts, MLCommon::Matrix::Data<double> **input,
             size_t n_rows, size_t n_cols, double *coef, double intercept,
             MLCommon::Matrix::Data<double> **preds, bool verbose);

};  // end namespace opg
};  // end namespace OLS
};  // end namespace ML
