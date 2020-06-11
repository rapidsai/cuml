/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <linalg/norm.cuh>
#include <linalg/unary_op.cuh>
#include <linalg/matrix_vector_op.cuh>

#include <common/device_buffer.hpp>

#include <cuml/common/cuml_allocator.hpp>


namespace MLCommon {
namespace Selection {

/**
 * @brief A virtual class defining pre- and post-processing
 * for metrics. This class will temporarily modify its given
 * state in `preprocess()` and undo those modifications in
 * `postprocess()`
 */

template<typename math_t>
class MetricProcessor {

public:
	virtual void preprocess(math_t *data) {}

	virtual void revert(math_t *data) {}

	virtual void postprocess(math_t *data) {}

	virtual ~MetricProcessor() = default;
};

template<typename math_t>
class CosineMetricProcessor : public MetricProcessor<math_t> {

public:

	CosineMetricProcessor(size_t n_rows, size_t n_cols, int k,
			bool row_major,
			cudaStream_t stream,
            std::shared_ptr<deviceAllocator> allocator) :
            	device_allocator_(allocator),
            	stream_(stream),
            	colsums_(allocator, stream, n_rows),
            	n_cols_(n_cols),
            	n_rows_(n_rows),
            	row_major_(row_major),
            	k_(k)
            {}

	void preprocess(math_t *data) {

		LinAlg::rowNorm(colsums_.data(), data, n_cols_, n_rows_, LinAlg::NormType::L2Norm,
		       row_major_, stream_, [] __device__(math_t in) {return sqrtf(in); });

		LinAlg::matrixVectorOp(data, data, colsums_.data(), n_cols_,
		               n_rows_, row_major_, true,
		               [=] __device__(math_t mat_in, math_t vec_in) {
							return mat_in / vec_in;
						}, stream_);
	}

	void revert(math_t *data) {
		LinAlg::matrixVectorOp(data, data, colsums_.data(), n_cols_,
		               n_rows_, row_major_, true,
		               [=] __device__(math_t mat_in, math_t vec_in) {
							return mat_in * vec_in;
						}, stream_);
	}

	void postprocess(math_t *data) {
		LinAlg::unaryOp(data, data, k_*n_rows_, [] __device__(math_t in) {return 1 - in; },
		             stream_);
	}

	~CosineMetricProcessor() = default;

	int k_;
	bool row_major_;
	size_t n_rows_;
	size_t n_cols_;
	cudaStream_t stream_;
	std::shared_ptr<deviceAllocator> device_allocator_;
	device_buffer<math_t> colsums_;
};

// Currently only being used by floats
template class MetricProcessor<float>;
template class CosineMetricProcessor<float>;

};
};
