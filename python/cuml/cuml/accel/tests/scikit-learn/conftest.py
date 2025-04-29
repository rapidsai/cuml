# Copyright (c) 2025, NVIDIA CORPORATION.
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


import logging
import os
from datetime import datetime
from functools import wraps

import pytest
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize NVML
try:
    nvmlInit()
    logger.info("NVML initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NVML: {e}")


def get_vram_usage():
    try:
        handle = nvmlDeviceGetHandleByIndex(
            0
        )  # Use device 0; customize if needed
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2  # in MB
    except Exception as e:
        logger.error(f"Error getting VRAM usage: {e}")
        return -1


def get_worker_id():
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def wrap_test_function(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        test_name = original_func.__name__
        worker = get_worker_id()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Running test: {test_name} [Worker: {worker}]")
        print(
            f"[{timestamp}] Running test: {test_name} [Worker: {worker}]",
            flush=True,
        )

        result = original_func(*args, **kwargs)

        vram = get_vram_usage()
        logger.info(
            f"[Worker: {worker}] [Test: {test_name}] VRAM Usage: {vram:.2f} MB"
        )
        print(
            f"[{timestamp}] [Worker: {worker}] [Test: {test_name}] VRAM Usage: {vram:.2f} MB",
            flush=True,
        )
        return result

    return wrapper


@pytest.hookimpl
def pytest_collection_modifyitems(session, config, items):
    # Wrap each test function to log VRAM usage
    for item in items:
        if hasattr(item, "obj"):  # Check if item is a test function
            item.obj = wrap_test_function(item.obj)


@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    try:
        nvmlShutdown()
        logger.info("NVML shutdown successfully")
    except Exception as e:
        logger.error(f"Error during NVML shutdown: {e}")
