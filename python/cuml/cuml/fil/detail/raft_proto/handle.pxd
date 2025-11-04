#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from pylibraft.common.handle cimport handle_t as raft_handle_t

from cuml.fil.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)


cdef extern from "cuml/fil/detail/raft_proto/handle.hpp" namespace "raft_proto" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(const raft_handle_t* handle_ptr) except +
        handle_t(const raft_handle_t& handle) except +
        raft_proto_stream_t get_next_usable_stream() except +
        void synchronize() except+
