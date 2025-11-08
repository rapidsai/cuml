#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
VALID_OUTPUT_TYPES = (
    "array",
    "numba",
    "dataframe",
    "series",
    "df_obj",
    "cupy",
    "numpy",
    "cudf",
    "pandas",
)

INTERNAL_VALID_OUTPUT_TYPES = ("input", *VALID_OUTPUT_TYPES)
