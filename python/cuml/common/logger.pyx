#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.string cimport string


cdef extern from "cuml/common/logger.hpp" namespace "ML" nogil:
    cdef cppclass Logger:
        pass


def set_level(level):
    Logger::get().setLevel(<int>level)


def set_pattern(pattern):
    cdef string s = pattern
    Logger::get().setPattern(s)


def should_log_for(level):
    return Logger::get().shouldLogFor(<int>level)


def get_pattern():
    return Logger::get().getPattern()
