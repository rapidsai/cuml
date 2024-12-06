#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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


import logging


IF GPUBUILD == 1:
    import sys

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


cdef class LogLevelSetter:
    """Internal "context manager" object for restoring previous log level"""

    def __cinit__(self, level_enum prev_log_level):
        self.prev_log_level = prev_log_level

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        IF GPUBUILD == 1:
            default_logger().set_level(self.prev_log_level)


def set_level(level):
    """
    Set logging level. This setting will be persistent from here onwards until
    the end of the process, if left unchanged afterwards.

    Examples
    --------

    .. code-block:: python

        # regular usage of setting a logging level for all subsequent logs
        # in this case, it will enable all logs upto and including `info()`
        logger.set_level(logger.level_enum.info)

        # in case one wants to temporarily set the log level for a code block
        with logger.set_level(logger.level_enum.debug) as _:
            logger.debug("Hello world!")

    Parameters
    ----------
    level : level_enum
        Logging level to be set.

    Returns
    -------
    context_object : LogLevelSetter
        This is useful if one wants to temporarily set a different logging
        level for a code section, as described in the example section above.
    """
    IF GPUBUILD == 1:
        cdef level_enum prev = default_logger().level()
        context_object = LogLevelSetter(prev)
        default_logger().set_level(level)
        return context_object


cdef class PatternSetter:
    """Internal "context manager" object for restoring previous log pattern"""

    def __init__(self, prev_pattern):
        self.prev_pattern = prev_pattern

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        IF GPUBUILD == 1:
            default_logger().set_pattern(self.prev_pattern)


def set_pattern(pattern):
    """
    Set the logging pattern. This setting will be persistent from here onwards
    until the end of the process, if left unchanged afterwards.

    Examples
    --------

    >>> # regular usage of setting a logging pattern for all subsequent logs
    >>> import cuml.internals.logger as logger
    >>> logger.set_pattern("--> [%H-%M-%S] %v")
    <cuml.internals.logger.PatternSetter object at 0x...>
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
    IF GPUBUILD == 1:
        # TODO: We probably can't implement this exact API because you can't
        # get the pattern from a spdlog logger since it could be different for
        # every sink (conversely, you could set because it forces every sink to
        # be the same). The best we can probably do is revert to the default
        # pattern.
        cdef string prev = default_pattern()
        # TODO: Need to cast to a Python string?
        context_object = PatternSetter(prev.decode("UTF-8"))
        cdef string s = pattern.encode("UTF-8")
        default_logger().set_pattern(s)
        return context_object


def should_log_for(level):
    """
    Check if messages at the given logging level will be logged or not. This is
    a useful check to avoid doing unnecessary logging work.

    Examples
    --------

    .. code-block:: python

        if logger.should_log_for(level_enum.info):
            # which could waste precious CPU cycles
            my_message = construct_message()
            logger.info(my_message)

    Parameters
    ----------
    level : level_enum
        Logging level to be set.
    """
    IF GPUBUILD == 1:
        return default_logger().should_log(level)


def _log(level_enum lvl, msg, default_func):
    """
    Internal function to log a message at a given level.

    Parameters
    ----------
    lvl : level_enum
        Logging level to be set.
    msg : str
        Message to be logged.
    default_func : function
        Default logging function to be used if GPU build is disabled.
    """
    IF GPUBUILD == 1:
        cdef string s = msg.encode("UTF-8")
        default_logger().log(lvl, s)
    ELSE:
        default_func(msg)


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
    # No trace level in Python so we use the closest thing, debug.
    _log(level_enum.trace, msg, logging.debug)


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
    _log(level_enum.debug, msg, logging.debug)


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
    _log(level_enum.info, msg, logging.info)


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
    _log(level_enum.warn, msg, logging.warn)


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
    _log(level_enum.error, msg, logging.error)


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
    _log(level_enum.critical, msg, logging.critical)


def flush():
    """
    Flush the logs.
    """
    IF GPUBUILD == 1:
        default_logger().flush()


IF GPUBUILD == 1:
    # Set callback functions to handle redirected sys.stdout in Python
    default_logger().sinks().push_back(<sink_ptr> make_shared[callback_sink_mt](_log_callback))
    # TODO: Expose flushing in the callback sink.
    # default_logger().setFlush(_log_flush)
