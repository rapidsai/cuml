/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/ensemble/treelite_defs.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

namespace ML {
namespace Explainer {

template <typename T>
class TreePathInfo;

using TreePathHandle =
  std::variant<std::shared_ptr<TreePathInfo<float>>, std::shared_ptr<TreePathInfo<double>>>;

using FloatPointer = std::variant<float*, double*>;

TreePathHandle extract_path_info(TreeliteModelHandle model);

void gpu_treeshap(TreePathHandle path_info,
                  const FloatPointer data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  FloatPointer out_preds,
                  std::size_t out_preds_size);

void gpu_treeshap_interventional(TreePathHandle path_info,
                                 const FloatPointer data,
                                 std::size_t n_rows,
                                 std::size_t n_cols,
                                 const FloatPointer background_data,
                                 std::size_t background_n_rows,
                                 std::size_t background_n_cols,
                                 FloatPointer out_preds,
                                 std::size_t out_preds_size);

void gpu_treeshap_interactions(TreePathHandle path_info,
                               const FloatPointer data,
                               std::size_t n_rows,
                               std::size_t n_cols,
                               FloatPointer out_preds,
                               std::size_t out_preds_size);

void gpu_treeshap_taylor_interactions(TreePathHandle path_info,
                                      const FloatPointer data,
                                      std::size_t n_rows,
                                      std::size_t n_cols,
                                      FloatPointer out_preds,
                                      std::size_t out_preds_size);

}  // namespace Explainer
}  // namespace ML
