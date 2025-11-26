#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper

import pytest

import cuml.internals.logger as logger


@pytest.mark.parametrize(
    "verbose, level",
    [
        (False, logger.level_enum.info),
        (True, logger.level_enum.debug),
        (-1, logger.level_enum.off),
        (0, logger.level_enum.off),
        (1, logger.level_enum.critical),
        (2, logger.level_enum.error),
        (3, logger.level_enum.warn),
        (4, logger.level_enum.info),
        (5, logger.level_enum.debug),
        (6, logger.level_enum.trace),
        (10, logger.level_enum.trace),
    ],
)
def test_verbose_to_level(verbose, level):
    assert logger._verbose_to_level(verbose) == level


def test_logger():
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warn("This is a warn message")
    logger.error("This is a error message")
    logger.critical("This is a critical message")

    with logger.set_level(logger.level_enum.warn):
        assert logger.should_log_for(logger.level_enum.warn)
        assert not logger.should_log_for(logger.level_enum.info)

    with logger.set_pattern("%v"):
        logger.info("This is an info message")


def test_set_level_get_level():
    orig = logger.get_level()
    assert isinstance(orig, logger.level_enum)

    with logger.set_level(logger.level_enum.trace):
        level = logger.get_level()
        assert isinstance(level, logger.level_enum)
        assert level == logger.level_enum.trace

    assert logger.get_level() == orig


def test_redirected_logger():
    new_stdout = StringIO()

    with logger.set_level(logger.level_enum.trace):
        # We do not test trace because CUML_LOG_TRACE is not compiled by
        # default
        test_msg = "This is a debug message"
        with redirect_stdout(new_stdout):
            logger.debug(test_msg)
        assert test_msg in new_stdout.getvalue()

        test_msg = "This is an info message"
        with redirect_stdout(new_stdout):
            logger.info(test_msg)
        assert test_msg in new_stdout.getvalue()

        test_msg = "This is a warn message"
        with redirect_stdout(new_stdout):
            logger.warn(test_msg)
        assert test_msg in new_stdout.getvalue()

        test_msg = "This is an error message"
        with redirect_stdout(new_stdout):
            logger.error(test_msg)
        assert test_msg in new_stdout.getvalue()

        test_msg = "This is a critical message"
        with redirect_stdout(new_stdout):
            logger.critical(test_msg)
        assert test_msg in new_stdout.getvalue()

    # Check that logging does not error with sys.stdout of None
    with redirect_stdout(None):
        test_msg = "This is a debug message"
        logger.debug(test_msg)


def test_log_flush():
    stdout_buffer = BytesIO()
    new_stdout = TextIOWrapper(stdout_buffer)

    with logger.set_level(logger.level_enum.trace):
        test_msg = "This is a debug message"
        with redirect_stdout(new_stdout):
            logger.debug(test_msg)
            assert test_msg not in stdout_buffer.getvalue().decode("utf-8")
            logger.flush()
            assert test_msg in stdout_buffer.getvalue().decode("utf-8")

    # Check that logging flush does not error with sys.stdout of None
    with redirect_stdout(None):
        logger.flush()
