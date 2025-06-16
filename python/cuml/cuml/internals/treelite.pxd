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
cdef extern from "treelite/c_api.h":
    ctypedef void* TreeliteModelHandle
    cdef int TreeliteSerializeModelToBytes(TreeliteModelHandle handle,
                                           const char** out_bytes, size_t* out_bytes_len)
    cdef int TreeliteDeserializeModelFromBytes(const char* bytes_seq, size_t len,
                                               TreeliteModelHandle* out)
    cdef int TreeliteFreeModel(TreeliteModelHandle handle)
    cdef const char* TreeliteGetLastError()
