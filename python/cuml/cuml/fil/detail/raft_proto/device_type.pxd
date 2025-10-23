#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "cuml/fil/detail/raft_proto/device_type.hpp" namespace "raft_proto" nogil:
    cdef enum device_type:
        cpu "raft_proto::device_type::cpu",
        gpu "raft_proto::device_type::gpu"
