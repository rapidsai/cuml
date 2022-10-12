#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

# distutils: language = c++


import sys

from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "cuml/common/logger.hpp" namespace "ML" nogil:
    cdef cppclass Logger:
        @staticmethod
        Logger& get()
        void setLevel(int level)
        void setPattern(const string& pattern)
        void setCallback(void(*callback)(int, char*))
        void setFlush(void(*flush)())
        bool shouldLogFor(int level) const
        int getLevel() const
        string getPattern() const
        void flush()


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
level_trace = CUML_LEVEL_TRACE

"""Enables all log messages upto and including `debug()`"""
level_debug = CUML_LEVEL_DEBUG

"""Enables all log messages upto and including `info()`"""
level_info = CUML_LEVEL_INFO

"""Enables all log messages upto and including `warn()`"""
level_warn = CUML_LEVEL_WARN

"""Enables all log messages upto and include `error()`"""
level_error = CUML_LEVEL_ERROR

"""Enables only `critical()` messages"""
level_critical = CUML_LEVEL_CRITICAL

"""Disables all log messages"""
level_off = CUML_LEVEL_OFF

cdef void _log_callback(int lvl, const char * msg) with gil:
    """
    Default spdlogs callback function to redirect logs correctly to sys.stdout

    Parameters
    ----------
    lvl : int
        Level of the logging message as defined by spdlogs
    msg : char *
        Message to be logged
    """
    print(msg.decode('utf-8'), end='')


cdef void _log_flush() with gil:
    """
    Default spdlogs callback function to flush logs
    """
    if sys.stdout is not None:
        sys.stdout.flush()


class LogLevelSetter:
    """Internal "context manager" object for restoring previous log level"""

    def __init__(self, prev_log_level):
        self.prev_log_level = prev_log_level

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        Logger.get().setLevel(<int>self.prev_log_level)


def set_level(level):
    """
    Set logging level. This setting will be persistent from here onwards until
    the end of the process, if left unchanged afterwards.

    Examples
    --------

    .. code-block:: python

        # regular usage of setting a logging level for all subsequent logs
        # in this case, it will enable all logs upto and including `info()`
        logger.set_level(logger.level_info)

        # in case one wants to temporarily set the log level for a code block
        with logger.set_level(logger.level_debug) as _:
            logger.debug("Hello world!")

    Parameters
    ----------
    level : int
        Logging level to be set. It must be one of cuml.common.logger.LEVEL_*

    Returns
    -------
    context_object : LogLevelSetter
        This is useful if one wants to temporarily set a different logging
        level for a code section, as described in the example section above.
    """
    cdef int prev = Logger.get().getLevel()
    context_object = LogLevelSetter(prev)
    Logger.get().setLevel(<int>level)
    return context_object


class PatternSetter:
    """Internal "context manager" object for restoring previous log pattern"""

    def __init__(self, prev_pattern):
        self.prev_pattern = prev_pattern

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        cdef string s = self.prev_pattern.encode("utf-8")
        Logger.get().setPattern(s)


def set_pattern(pattern):
    """
    Set the logging pattern. This setting will be persistent from here onwards
    until the end of the process, if left unchanged afterwards.

    Examples
    --------

    >>> # regular usage of setting a logging pattern for all subsequent logs
    >>> import cuml.common.logger as logger
    >>> logger.set_pattern("--> [%H-%M-%S] %v")
    <cuml.common.logger.PatternSetter object at 0x...>
    >>> # in case one wants to temporarily set the pattern for a code block
    >>> with logger.set_pattern("--> [%H-%M-%S] %v") as _:
    ...     logger.info("Hello world!")
    --> [...] Hello world!

    Parameters
    ----------
    pattern : str
        Logging pattern string. Refer to this wiki page for its syntax:
        https://github.com/gabime/spdlog/wiki/3.-Custom-formatting

    Returns
    -------
    context_object : PatternSetter
        This is useful if one wants to temporarily set a different logging
        pattern for a code section, as described in the example section above.
    """
    cdef string prev = Logger.get().getPattern()
    context_object = PatternSetter(prev.decode("UTF-8"))
    cdef string s = pattern.encode("UTF-8")
    Logger.get().setPattern(s)
    return context_object


def should_log_for(level):
    """
    Check if messages at the given logging level will be logged or not. This is
    a useful check to avoid doing unnecessary logging work.

    Examples
    --------

    .. code-block:: python

        if logger.should_log_for(level_info):
            # which could waste precious CPU cycles
            my_message = construct_message()
            logger.info(my_message)

    Parameters
    ----------
    level : int
        Logging level to be set. It must be one of cuml.common.logger.level_*
    """
    return Logger.get().shouldLogFor(<int>level)


def trace(msg):
    """
    Logs a trace message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.trace("Hello world! This is a trace message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_TRACE(s.c_str())


def debug(msg):
    """
    Logs a debug message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.debug("Hello world! This is a debug message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_DEBUG(s.c_str())


def info(msg):
    """
    Logs an info message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.info("Hello world! This is a info message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_INFO(s.c_str())


def warn(msg):
    """
    Logs a warning message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.warn("Hello world! This is a warning message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_WARN(s.c_str())


def error(msg):
    """
    Logs an error message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.error("Hello world! This is a error message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_ERROR(s.c_str())


def critical(msg):
    """
    Logs a critical message, if it is enabled.

    Examples
    --------

    .. code-block:: python

                logger.critical("Hello world! This is a critical message")

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    CUML_LOG_CRITICAL(s.c_str())


def flush():
    """
    Flush the logs.
    """
    Logger.get().flush()


# Set callback functions to handle redirected sys.stdout in Python
Logger.get().setCallback(_log_callback)
Logger.get().setFlush(_log_flush)
