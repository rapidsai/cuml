#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

import cuml.common.opg_data_utils_mg as opg
import cuml.internals
from cuml.common import input_to_cuml_array


class BaseDecompositionMG:
    @cuml.internals.api_base_return_any_skipall
    def fit(self, X, total_rows, n_cols, partsToRanks, rank, _transform=False):
        """
        Fit function for PCA/TSVD MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param total_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """
        self._set_output_type(X[0])
        self._set_n_features_in(n_cols)

        if self.n_components is None:
            self.n_components_ = min(total_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        X_arys = []
        dtype = None
        for i in range(len(X)):
            X_m, _, self.n_cols, dtype = input_to_cuml_array(
                X[i],
                check_dtype=(
                    [np.float32, np.float64] if dtype is None else dtype
                ),
            )
            X_arys.append(X_m)

        X_arg = opg.build_data_t(X_arys)

        rank_to_sizes = opg.build_rank_size_pair(partsToRanks, rank)

        input_desc = opg.build_part_descriptor(
            total_rows, self.n_cols, rank_to_sizes, rank
        )

        if _transform:
            trans_arys = opg.build_pred_or_trans_arys(X_arys, "F", dtype)
            trans_arg = opg.build_data_t(trans_arys)
            trans_desc = opg.build_part_descriptor(
                total_rows, self.n_components_, rank_to_sizes, rank
            )

        if _transform:
            self._mg_fit_transform(
                X_arg,
                total_rows,
                n_cols,
                dtype,
                trans_arg,
                input_desc,
                trans_desc,
            )
        else:
            self._mg_fit(X_arg, total_rows, n_cols, dtype, input_desc)

        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(input_desc)
        opg.free_data_t(X_arg, dtype)

        if _transform:
            trans_out = []

            for i in range(len(trans_arys)):
                trans_out.append(
                    trans_arys[i].to_output(
                        output_type=self._get_output_type(X[0])
                    )
                )

            opg.free_data_t(trans_arg, dtype)
            opg.free_part_descriptor(trans_desc)

            return trans_out
