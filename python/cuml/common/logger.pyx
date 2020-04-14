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
from libcpp cimport bool


cdef extern from "cuml/common/logger.hpp" namespace "ML" nogil:
    cdef cppclass Logger:
        @staticmethod
        Logger& get()
        void setLevel(int level)
        void setPattern(const string& pattern)
        bool shouldLogFor(int level) const
        string getPattern() const

cdef extern from "cuml/common/logger.hpp" nogil:
    void CUML_LOG_TRACE(const char* fmt, ...)
    void CUML_LOG_DEBUG(const char* fmt, ...)
    void CUML_LOG_INFO(const char* fmt, ...)
    void CUML_LOG_WARN(const char* fmt, ...)
    void CUML_LOG_ERROR(const char* fmt, ...)
    void CUML_LOG_CRITICAL(const char* fmt, ...)

    cdef int CUML_LEVEL_TRACE
    cdef int CUML_LEVEL_DEBUG
    cdef int CUML_LEVEL_INFO
    cdef int CUML_LEVEL_WARN
    cdef int CUML_LEVEL_ERROR
    cdef int CUML_LEVEL_CRITICAL
    cdef int CUML_LEVEL_OFF


"""Enables all log messages upto and including `trace()`"""
LEVEL_TRACE = CUML_LEVEL_TRACE

"""Enables all log messages upto and including `debug()`"""
LEVEL_DEBUG = CUML_LEVEL_DEBUG

"""Enables all log messages upto and including `info()`"""
LEVEL_INFO = CUML_LEVEL_INFO

"""Enables all log messages upto and including `warn()`"""
LEVEL_WARN = CUML_LEVEL_WARN

"""Enables all log messages upto and include `error()`"""
LEVEL_ERROR = CUML_LEVEL_ERROR

"""Enables only `critical()` messages"""
LEVEL_CRITICAL = CUML_LEVEL_CRITICAL

"""Disables all log messages"""
LEVEL_OFF = CUML_LEVEL_OFF


def set_level(level):
    """
    Set logging level. This setting will be persistent from here onwards until
    the end of the process, if left unchanged afterwards.

    Examples
    --------

    .. code-block:: python

        # To enable all log messages upto and including `info()`
        import cuml.common.logger as logger
        logger.set_level(logger.LEVEL_INFO)

    Parameters
    ----------
    level : int
        Logging level to be set. It must be one of cuml.common.logger.LEVEL_*
    """
    Logger.get().setLevel(<int>level)


def set_pattern(pattern):
    """
    Set the logging pattern. This setting will be persistent from here onwards
    until the end of the process, if left unchanged afterwards.

    Examples
    --------

    .. code-block:: python

        import cuml.common.logger as logger
        logger.set_pattern("--> [%H-%M-%S] %v")

    Parameters
    ----------
    pattern : str
        Logging pattern string. Refer to this wiki page for its syntax:
        https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    """
    cdef string s = pattern
    Logger.get().setPattern(s)


def should_log_for(level):
    """
    Check if messages at the given logging level will be logged or not. This is
    a useful check to avoid doing unnecessary logging work.

    Examples
    --------

    .. code-block:: python

        import cuml.common.logger as logger
        if logger.should_log_for(LEVEL_INFO):
            # which could waste precious CPU cycles
            my_message = construct_message()
            logger.info(my_message)

    Parameters
    ----------
    level : int
        Logging level to be set. It must be one of cuml.common.logger.LEVEL_*
    """
    return Logger.get().shouldLogFor(<int>level)


def get_pattern():
    """
    Returns the current logging pattern. Useful in case one is temporarily
    changing the pattern, like in a method.

    Examples
    --------

    .. code-block:: python

        import cuml.common.logger as logger
        def some_func(new_patt):
            old_patt = logger.get_pattern()
            logger.set_pattern(new_patt)
            do_work()
            logger.set_pattern(old_patt)
    """
    cdef string s = Logger.get().getPattern()
    return s.decode("UTF-8")


def trace(msg):
    cdef string s = msg
    CUML_LOG_TRACE(s.c_str())


def debug(msg):
    cdef string s = msg
    CUML_LOG_DEBUG(s.c_str())


def info(msg):
    cdef string s = msg
    CUML_LOG_INFO(s.c_str())


def warn(msg):
    cdef string s = msg
    CUML_LOG_WARN(s.c_str())


def error(msg):
    cdef string s = msg
    CUML_LOG_ERROR(s.c_str())


def critical(msg):
    cdef string s = msg
    CUML_LOG_CRITICAL(s.c_str())
