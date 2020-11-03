#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

import _pytest.config
import _pytest.python
import _pytest.terminal
import cupy as cp
import pytest
import rmm
import rmm._lib

# rmm.reinitialize(logging=True, log_file_name="test_log.txt")

tracked_mr = rmm.mr.TrackedResourceAdaptor(
    rmm.mr.get_current_device_resource())

rmm.mr.set_current_device_resource(tracked_mr)

test_memory_info = {
    "total": {
        "count": 0, "peak": 0, "nbytes": 0, "streams": {
            0: 0,
        }
    }
}


# Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is not used
def bad_allocator(nbytes):

    assert False, \
        "Using default cupy allocator instead of rmm.rmm_cupy_allocator"

    return None


saved_allocator = rmm.rmm_cupy_allocator


def counting_rmm_allocator(nbytes):

    import cuml.common.array

    cuml.common.array._increment_malloc(nbytes)

    return saved_allocator(nbytes)


rmm.rmm_cupy_allocator = counting_rmm_allocator


def pytest_configure(config):
    cp.cuda.set_allocator(counting_rmm_allocator)


@pytest.fixture(scope="function", autouse=True)
def cupy_allocator_fixture(request):

    # Disable creating cupy arrays
    # cp.cuda.set_allocator(bad_allocator)
    cp.cuda.set_allocator(counting_rmm_allocator)

    allocations = {}
    memory = {
        "outstanding": 0,
        "peak": 0,
        "count": 0,
        "nbytes": 0,
        "streams": {
            0: 0,
        }
    }

    def print_mr(is_alloc: bool, mem_ptr, n_bytes: int, stream_ptr):

        if (stream_ptr not in memory["streams"]):
            memory["streams"][stream_ptr] = 0

        if (is_alloc):
            assert mem_ptr not in allocations
            allocations[mem_ptr] = n_bytes
            memory["outstanding"] += n_bytes
            memory["peak"] = max(memory["outstanding"], memory["peak"])
            memory["count"] += 1
            memory["nbytes"] += n_bytes
            memory["streams"][stream_ptr] += n_bytes
        else:
            # assert mem_ptr in allocations
            popped_nbytes = allocations.pop(mem_ptr, 0)
            memory["outstanding"] -= n_bytes if n_bytes > 0 else popped_nbytes

    # callback_mr.set_callback(print_mr)

    tracked_mr.reset_info()

    yield

    import gc

    gc.collect()

    alloc_info = tracked_mr.get_info()

    # assert len(allocations) == 0

    # del memory["outstanding"]
    alloc_info["streams"] = {0: alloc_info["nbytes"]}

    test_memory_info[request.node.nodeid] = alloc_info

    test_memory_info["total"].update({
        "count": test_memory_info["total"]["count"] + alloc_info["count"],
        "peak": max(test_memory_info["total"]["peak"], alloc_info["peak"]),
        "nbytes": test_memory_info["total"]["nbytes"] + alloc_info["nbytes"],
    })

    test_memory_info["total"]["streams"].update({
        0: test_memory_info["total"]["streams"][0] + alloc_info["streams"][0],
    })

    # Reset creating cupy arrays
    cp.cuda.set_allocator(None)


def pytest_terminal_summary(
        terminalreporter: _pytest.terminal.TerminalReporter,
        exitstatus: pytest.ExitCode,
        config: _pytest.config.Config):

    terminalreporter.write_sep("=", "CumlArray Summary")

    import cuml.common.array

    terminalreporter.write_line("To Output Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._to_output_counts))

    terminalreporter.write_line("From Array Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._from_array_counts))

    terminalreporter.write_line("CuPy Malloc: Count={}, Size={}".format(
        cuml.common.array._malloc_count.get(),
        cuml.common.array._malloc_nbytes.get()))

    have_outstanding = list(
        filter(lambda x: "outstanding" in x[1] and x[1]["outstanding"] > 0,
               test_memory_info.items()))

    if (len(have_outstanding) > 0):
        terminalreporter.write_line("Memory leak in the following tests:",
                                    red=True)

        for key, memory in have_outstanding:
            terminalreporter.write_line(key)

    terminalreporter.write_line("Allocation Info: (test, peak, count)",
                                yellow=True)

    count = 0

    for key, memory in sorted(test_memory_info.items(),
                              key=lambda x: -x[1]["peak"]):

        default_stream_nbytes = (memory["streams"][0] / memory["nbytes"]
                                 if memory["nbytes"] > 0 else 1.0)

        terminalreporter.write_line(
            ("Peak={:>12,}, NBytes={:>12,}, Count={:>6,}"
             ", Stream0={:>6.1%}, Test={}").format(memory["peak"],
                                                   memory["nbytes"],
                                                   memory["count"],
                                                   default_stream_nbytes,
                                                   key))

        # if (count > 50):
        #     break

        count += 1
