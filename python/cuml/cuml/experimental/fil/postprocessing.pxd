#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
cdef extern from "cuml/experimental/fil/postproc_ops.hpp" namespace "ML::experimental::fil" nogil:
    cdef enum row_op:
        row_disable "ML::experimental::fil::row_op::disable",
        softmax "ML::experimental::fil::row_op::softmax",
        max_index "ML::experimental::fil::row_op::max_index"
    cdef enum element_op:
        elem_disable "ML::experimental::fil::element_op::disable",
        signed_square "ML::experimental::fil::element_op::signed_square",
        hinge "ML::experimental::fil::element_op::hinge",
        sigmoid "ML::experimental::fil::element_op::sigmoid",
        exponential "ML::experimental::fil::element_op::exponential",
        logarithm_one_plus_exp "ML::experimental::fil::element_op::logarithm_one_plus_exp"
