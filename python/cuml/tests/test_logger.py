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

from contextlib import redirect_stdout
import cuml.internals.logger as logger
from io import StringIO, TextIOWrapper, BytesIO


def test_logger():
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warn("This is a warn message")
    logger.error("This is a error message")
    logger.critical("This is a critical message")

    with logger.set_level(logger.level_warn):
        assert(logger.should_log_for(logger.level_warn))
        assert(not logger.should_log_for(logger.level_info))

    with logger.set_pattern("%v"):
        logger.info("This is an info message")


def test_redirected_logger():
    new_stdout = StringIO()

    with logger.set_level(logger.level_trace):
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

    with logger.set_level(logger.level_trace):
        test_msg = "This is a debug message"
        with redirect_stdout(new_stdout):
            logger.debug(test_msg)
            assert test_msg not in stdout_buffer.getvalue().decode('utf-8')
            logger.flush()
            assert test_msg in stdout_buffer.getvalue().decode('utf-8')

    # Check that logging flush does not error with sys.stdout of None
    with redirect_stdout(None):
        logger.flush()
