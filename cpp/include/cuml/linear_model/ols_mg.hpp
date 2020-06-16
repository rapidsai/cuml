/*
* Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
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
