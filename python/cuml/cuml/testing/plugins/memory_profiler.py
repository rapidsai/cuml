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
#

import warnings

import pytest
from rmm.statistics import get_statistics, statistics


class HighMemoryUsageWarning(UserWarning):
    """Warning emitted when a test exceeds the memory usage threshold."""

    pass


# Memory threshold in MB for reporting memory usage
MEMORY_REPORT_THRESHOLD_MB = 1024


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap test execution with GPU memory profiler."""
    with statistics():
        yield

        # Check memory usage after test completion
        stats = get_statistics()
        peak_memory_mb = stats.peak_bytes / (1024 * 1024)

        if peak_memory_mb > MEMORY_REPORT_THRESHOLD_MB:
            msg = (
                f"Test {item.nodeid} used {peak_memory_mb:.2f} MB of GPU memory, "
                f"exceeding threshold of {MEMORY_REPORT_THRESHOLD_MB} MB"
            )
            warnings.warn(msg, HighMemoryUsageWarning)
