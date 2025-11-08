#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "treelite/c_api.h":
    ctypedef void* TreeliteModelHandle
    cdef int TreeliteSerializeModelToBytes(TreeliteModelHandle handle,
                                           const char** out_bytes, size_t* out_bytes_len)
    cdef int TreeliteDeserializeModelFromBytes(const char* bytes_seq, size_t len,
                                               TreeliteModelHandle* out)
    cdef int TreeliteFreeModel(TreeliteModelHandle handle)
    cdef const char* TreeliteGetLastError()
