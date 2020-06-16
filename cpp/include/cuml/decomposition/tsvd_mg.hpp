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
