#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuml/fil/infer_kind.hpp" namespace "ML::fil":
    # TODO(hcho3): Switch to new syntax for scoped enum when we adopt Cython 3.0
    cdef enum infer_kind:
        default_kind "ML::fil::infer_kind::default_kind"
        per_tree "ML::fil::infer_kind::per_tree"
        leaf_id "ML::fil::infer_kind::leaf_id"
