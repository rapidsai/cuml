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

import pytest

from cuml.common.logger import logger


def test_logger():
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is a info message")
    logger.warn("This is a warn message")
    logger.error("This is a error message")
    logger.critical("This is a critical message")

    with logger.set_level(logger.LOG_LEVEL_WARN):
        assert(logger.should_log_for(logger.LOG_LEVEL_WARN))
        assert(not logger.should_log_for(logger.LOG_LEVEL_INFO))
