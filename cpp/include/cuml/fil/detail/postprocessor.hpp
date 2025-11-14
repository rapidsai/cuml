/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/postproc_ops.hpp>

#include <stddef.h>

#include <limits>
#include <type_traits>

#ifndef __CUDACC__
#include <math.h>
#endif

namespace ML {
namespace fil {

/* Convert the postprocessing operations into a single value
 * representing what must be done in the inference kernel
 */
HOST DEVICE inline auto constexpr ops_to_val(row_op row_wise, element_op elem_wise)
{
  return (static_cast<std::underlying_type_t<row_op>>(row_wise) |
          static_cast<std::underlying_type_t<element_op>>(elem_wise));
}

/*
 * Perform postprocessing on raw forest output
 *
 * @param val Pointer to the raw forest output
 * @param output_count The number of output values per row
 * @param bias Pointer to bias vector, which is added to the output
 * as part of the postprocessing step
 * @param out Pointer to the output buffer
 * @param stride Number of elements between the first element that must be
 * summed for a particular output element and the next. This is typically
 * equal to the number of "groves" of trees over which the computation
 * was divided.
 * @param average_factor The factor by which to divide during the
 * normalization step of postprocessing
 * @param constant If the postprocessing operation requires a constant,
 * it can be passed here.
 */
template <row_op row_wise_v, element_op elem_wise_v, typename io_t>
HOST DEVICE void postprocess(infer_kind infer_type,
                             io_t* val,
                             index_type output_count,
                             const io_t* bias,
                             io_t* out,
                             index_type stride   = index_type{1},
                             io_t average_factor = io_t{1},
                             io_t constant       = io_t{1})
{
  const bool use_bias = infer_type == infer_kind::default_kind;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
  auto max_index = index_type{};
  auto max_value = std::numeric_limits<io_t>::lowest();
#pragma GCC diagnostic pop
  for (auto output_index = index_type{}; output_index < output_count; ++output_index) {
    auto workspace_index = output_index * stride;
    // Add the bias term if use_bias is true.
    // The following expression is written to avoid branching.
    val[workspace_index] =
      val[workspace_index] / average_factor +
      bias[output_index * static_cast<index_type>(use_bias)] * static_cast<io_t>(use_bias);
    if constexpr (elem_wise_v == element_op::signed_square) {
      val[workspace_index] =
        copysign(val[workspace_index] * val[workspace_index], val[workspace_index]);
    } else if constexpr (elem_wise_v == element_op::hinge) {
      val[workspace_index] = io_t(val[workspace_index] > io_t{});
    } else if constexpr (elem_wise_v == element_op::sigmoid) {
      val[workspace_index] = io_t{1} / (io_t{1} + exp(-constant * val[workspace_index]));
    } else if constexpr (elem_wise_v == element_op::exponential) {
      val[workspace_index] = exp(val[workspace_index] / constant);
    } else if constexpr (elem_wise_v == element_op::logarithm_one_plus_exp) {
      val[workspace_index] = log1p(exp(val[workspace_index] / constant));
    }
    if constexpr (row_wise_v == row_op::softmax || row_wise_v == row_op::max_index) {
      auto is_new_max = val[workspace_index] > max_value;
      max_index       = is_new_max * output_index + (!is_new_max) * max_index;
      max_value       = is_new_max * val[workspace_index] + (!is_new_max) * max_value;
    }
  }

  if constexpr (row_wise_v == row_op::max_index) {
    *out = max_index;
  } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    auto softmax_normalization = io_t{};
#pragma GCC diagnostic pop
    if constexpr (row_wise_v == row_op::softmax) {
      for (auto workspace_index = index_type{}; workspace_index < output_count * stride;
           workspace_index += stride) {
        val[workspace_index] = exp(val[workspace_index] - max_value);
        softmax_normalization += val[workspace_index];
      }
    }

    for (auto output_index = index_type{}; output_index < output_count; ++output_index) {
      auto workspace_index = output_index * stride;
      if constexpr (row_wise_v == row_op::softmax) {
        out[output_index] = val[workspace_index] / softmax_normalization;
      } else {
        out[output_index] = val[workspace_index];
      }
    }
  }
}

/*
 * Struct which holds all data necessary to perform postprocessing on raw
 * output of a forest model
 *
 * @tparam io_t The type used for input and output to/from the model
 * (typically float/double)
 * @param row_wise Enum value representing the row-wise post-processing
 * operation to perform on the output
 * @param elem_wise Enum value representing the element-wise post-processing
 * operation to perform on the output
 * @param average_factor The factor by which to divide during the
 * normalization step of postprocessing
 * @param constant If the postprocessing operation requires a constant,
 * it can be passed here.
 */
template <typename io_t>
struct postprocessor {
  HOST DEVICE postprocessor(row_op row_wise      = row_op::disable,
                            element_op elem_wise = element_op::disable,
                            io_t average_factor  = io_t{1},
                            io_t constant        = io_t{1})
    : average_factor_{average_factor},
      constant_{constant},
      row_wise_{row_wise},
      elem_wise_{elem_wise}
  {
  }

  HOST DEVICE void operator()(infer_kind infer_type,
                              io_t* val,
                              index_type output_count,
                              const io_t* bias,
                              io_t* out,
                              index_type stride = index_type{1}) const
  {
    switch (ops_to_val(row_wise_, elem_wise_)) {
      case ops_to_val(row_op::disable, element_op::signed_square):
        postprocess<row_op::disable, element_op::signed_square>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::disable, element_op::hinge):
        postprocess<row_op::disable, element_op::hinge>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::disable, element_op::sigmoid):
        postprocess<row_op::disable, element_op::sigmoid>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::disable, element_op::exponential):
        postprocess<row_op::disable, element_op::exponential>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::disable, element_op::logarithm_one_plus_exp):
        postprocess<row_op::disable, element_op::logarithm_one_plus_exp>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::disable):
        postprocess<row_op::softmax, element_op::disable>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::signed_square):
        postprocess<row_op::softmax, element_op::signed_square>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::hinge):
        postprocess<row_op::softmax, element_op::hinge>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::sigmoid):
        postprocess<row_op::softmax, element_op::sigmoid>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::exponential):
        postprocess<row_op::softmax, element_op::exponential>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::softmax, element_op::logarithm_one_plus_exp):
        postprocess<row_op::softmax, element_op::logarithm_one_plus_exp>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::disable):
        postprocess<row_op::max_index, element_op::disable>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::signed_square):
        postprocess<row_op::max_index, element_op::signed_square>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::hinge):
        postprocess<row_op::max_index, element_op::hinge>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::sigmoid):
        postprocess<row_op::max_index, element_op::sigmoid>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::exponential):
        postprocess<row_op::max_index, element_op::exponential>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      case ops_to_val(row_op::max_index, element_op::logarithm_one_plus_exp):
        postprocess<row_op::max_index, element_op::logarithm_one_plus_exp>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
        break;
      default:
        postprocess<row_op::disable, element_op::disable>(
          infer_type, val, output_count, bias, out, stride, average_factor_, constant_);
    }
  }

 private:
  io_t average_factor_;
  io_t constant_;
  row_op row_wise_;
  element_op elem_wise_;
};
}  // namespace fil
}  // namespace ML
