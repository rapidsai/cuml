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
from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper

import pytest

import cuml.internals.logger as logger


@pytest.mark.parametrize(
    "verbose, level, verbose_numeric",
    [
        (False, logger.level_enum.info, 4),
        (True, logger.level_enum.debug, 5),
        (0, logger.level_enum.off, 0),
        (1, logger.level_enum.critical, 1),
        (2, logger.level_enum.error, 2),
        (3, logger.level_enum.warn, 3),
        (4, logger.level_enum.info, 4),
        (5, logger.level_enum.debug, 5),
        (6, logger.level_enum.trace, 6),
    ],
)
def test_verbose_to_from_level(verbose, level, verbose_numeric):
    assert logger._verbose_to_level(verbose) == level
    assert logger._verbose_from_level(level) == verbose_numeric


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
