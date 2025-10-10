#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.string cimport string


cdef extern from "rapids_logger/logger.hpp" namespace "rapids_logger" nogil:

    cpdef enum class level_enum:
        trace
        debug
        info
        warn
        error
        critical
        off
        n_levels

    cdef cppclass sink:
        pass

    ctypedef shared_ptr[sink] sink_ptr

# Spoof the logger as a namespace to get the sink_vector generated correctly.
cdef extern from "rapids_logger/logger.hpp" namespace "rapids_logger::logger" nogil:

    cdef cppclass sink_vector:
        void push_back(const sink_ptr& sink) except +
        void pop_back() except +
        void clear()

cdef extern from "rapids_logger/logger.hpp" namespace "rapids_logger" nogil:
    cdef cppclass logger:
        logger(string name, string filename) except +
        void set_level(level_enum log_level) except +
        void set_pattern(const string& pattern)
        level_enum level() except +
        void flush() except +
        void flush_on(level_enum level) except +
        level_enum flush_level() except +
        bool should_log(level_enum msg_level) except +
        void log(level_enum lvl, const string& fmt, ...)
        const sink_vector& sinks() const

    ctypedef void(*log_callback_t)(int, const char*) except * with gil
    ctypedef void(*flush_callback_t)() except * with gil

    cdef cppclass callback_sink_mt:
        callback_sink_mt(log_callback_t callback, flush_callback_t flush) except +

cdef extern from "cuml/common/logger.hpp" namespace "ML" nogil:
    cdef logger& default_logger() except +
    cdef string default_pattern() except +

cdef void _log_callback(int lvl, const char * msg) with gil
cdef void _log_flush() with gil


cdef class LogLevelSetter:
    """Internal "context manager" object for restoring previous log level"""
    cdef level_enum prev_log_level


cdef class PatternSetter:
    """Internal "context manager" object for restoring previous log pattern"""
    cdef string prev_pattern
