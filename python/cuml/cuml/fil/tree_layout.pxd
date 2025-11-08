#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "cuml/fil/tree_layout.hpp" namespace "ML::fil" nogil:
    cdef enum tree_layout:
        depth_first "ML::fil::tree_layout::depth_first",
        breadth_first "ML::fil::tree_layout::breadth_first",
        layered_children_together "ML::fil::tree_layout::layered_children_together"
