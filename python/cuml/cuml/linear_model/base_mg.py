#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cuml.common.opg_data_utils_mg as opg
from cuml.internals import mlfunc
from cuml.internals.validation import check_inputs


class MGFitMixin:
    def __init__(self, *, handle, **kwargs):
        self.handle = handle
        super().__init__(**kwargs)

    @mlfunc(convert_output=False)
    def fit(self, input_data, n_rows, n_cols, parts_rank_size, rank):
        """
        Fit function for MNMG linear regression classes

        This not meant to be used as part of the public API.

        Parameters
        ----------
        X : list of array-like
            A list of array-like partitions to use as training data.
        y : list of array-like
            A list of array-like partitions to use as target values.
        n_rows : int
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
        self._set_output_type(input_data[0][0])

        Xs = []
        ys = []
        dtype = ("float32", "float64")
        for i, (X, y) in enumerate(input_data):
            X, y = check_inputs(
                self,
                X,
                y,
                dtype=dtype,
                order="F",
                reset=(i == 0),
            )
            if i == 0:
                dtype = X.dtype

            Xs.append(X)
            ys.append(y)

        rank_to_sizes = opg.build_rank_size_pair(parts_rank_size)
        input_desc_ptr = opg.build_part_descriptor(
            n_rows, n_cols, rank_to_sizes, rank
        )
        X_ptr = opg.build_data_t(Xs)
        y_ptr = opg.build_data_t(ys)

        self._fit(X_ptr, y_ptr, n_cols, dtype, input_desc_ptr)

        opg.free_data_t(X_ptr, dtype)
        opg.free_data_t(y_ptr, dtype)
        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(input_desc_ptr)

        return self
