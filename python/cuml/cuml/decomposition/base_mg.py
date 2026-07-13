#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

import cuml.common.opg_data_utils_mg as opg
from cuml.internals.outputs import convert_arrays, mlfunc
from cuml.internals.validation import check_inputs


class BaseDecompositionMG:
    def __init__(self, *, handle, **kwargs):
        self.handle = handle
        super().__init__(**kwargs)

    @mlfunc(convert_output=False)
    def fit(
        self, X, total_rows, n_cols, parts_rank_size, rank, _transform=False
    ):
        """
        Fit function for PCA/TSVD MG.

        This not meant to be used as part of the public API.

        Parameters
        ----------
        X : list of array-like
            A list of array-like partitions to use as training data.
        total_rows : int
            The total number of rows in X across all distributed nodes.
        n_cols : int
            The number of columns in X.
        parts_rank_size : list[tuple[int, int]]
            A list of tuples of (rank, size)
        rank : int
            The current rank.

        Returns
        -------
        self
        """
        self._set_output_type(X[0])

        if self.n_components is None:
            self.n_components_ = min(total_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        Xs = []
        dtype = ("float32", "float64")
        for i, X_part in enumerate(X):
            X_part = check_inputs(
                self,
                X_part,
                dtype=dtype,
                order="F",
                reset=(i == 0),
            )
            if i == 0:
                dtype = X_part.dtype
            Xs.append(X_part)

        X_ptr = opg.build_data_t(Xs)

        rank_to_sizes = opg.build_rank_size_pair(parts_rank_size)
        input_desc_ptr = opg.build_part_descriptor(
            total_rows, n_cols, rank_to_sizes, rank
        )

        if _transform:
            trans_arys = [
                cp.zeros(
                    (X.shape[0], self.n_components), dtype=dtype, order="F"
                )
                for X in Xs
            ]
            trans_ptr = opg.build_data_t(trans_arys)
            trans_desc_ptr = opg.build_part_descriptor(
                total_rows, self.n_components_, rank_to_sizes, rank
            )

        if _transform:
            self._mg_fit_transform(
                X_ptr,
                total_rows,
                n_cols,
                dtype,
                trans_ptr,
                input_desc_ptr,
                trans_desc_ptr,
            )
        else:
            self._mg_fit(X_ptr, total_rows, n_cols, dtype, input_desc_ptr)

        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(input_desc_ptr)
        opg.free_data_t(X_ptr, dtype)

        if _transform:
            output_type = self._get_output_type(X[0])
            trans_out = [
                convert_arrays(part, output_type=output_type)
                for part in trans_arys
            ]

            opg.free_data_t(trans_ptr, dtype)
            opg.free_part_descriptor(trans_desc_ptr)

            return trans_out
