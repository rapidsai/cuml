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

import os
import psutil

# Memory threshold in MB for reporting memory usage
MEMORY_REPORT_THRESHOLD_MB = 1024


def get_process_memory():
    """Get the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


class MemoryProfiler:
    def __init__(self):
        self.start_memory = None
        self.max_memory = 0

    def pytest_runtest_setup(self, item):
        """Record memory usage at test setup."""
        self.start_memory = get_process_memory()

    def pytest_runtest_teardown(self, item):
        """Record memory usage at test teardown and report if significant."""
        end_memory = get_process_memory()
        if self.start_memory is not None:
            memory_used = end_memory - self.start_memory
            self.max_memory = max(self.max_memory, end_memory)
            if memory_used > MEMORY_REPORT_THRESHOLD_MB:
                print(f"\nMemory usage for {item.nodeid}:")
                print(f"  Start: {self.start_memory:.2f} MB")
                print(f"  End: {end_memory:.2f} MB")
                print(f"  Delta: {memory_used:.2f} MB")
                print(f"  Max: {self.max_memory:.2f} MB")


def pytest_configure(config):
    """Register the memory profiler plugin."""
    config.pluginmanager.register(MemoryProfiler())
