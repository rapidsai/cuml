#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "cuml/fil/detail/raft_proto/cuda_stream.hpp" namespace "raft_proto" nogil:
    cdef cppclass cuda_stream:
        pass
