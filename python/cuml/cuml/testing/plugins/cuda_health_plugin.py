#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Pytest plugin to monitor CUDA health during test execution.

This plugin helps identify tests that cause CUDA memory corruption by:
1. Checking GPU health before and after each test
2. Logging CUDA memory state
3. Detecting illegal memory access errors early
4. Providing detailed reports on which test caused issues

Usage:
    pytest --cuda-health-check
    pytest --cuda-health-check --cuda-health-verbose
    pytest --cuda-health-check --cuda-health-sync
    pytest --cuda-health-check --cuda-health-gc

With xdist, logs go to cuda_health_<worker>.log files.
"""

import os
import sys
import time

import pytest


def pytest_addoption(parser):
    """Add command line options for CUDA health monitoring."""
    group = parser.getgroup("CUDA Health Monitoring")

    group.addoption(
        "--cuda-health-check",
        action="store_true",
        default=False,
        help="Enable CUDA health checking between tests",
    )

    group.addoption(
        "--cuda-health-verbose",
        action="store_true",
        default=False,
        help="Verbose CUDA health logging",
    )

    group.addoption(
        "--cuda-health-sync",
        action="store_true",
        default=False,
        help="Force CUDA stream sync after each test",
    )

    group.addoption(
        "--cuda-health-gc",
        action="store_true",
        default=False,
        help="Force garbage collection after each test",
    )


class CUDAHealthPlugin:
    """Plugin to monitor CUDA health during test execution."""

    def __init__(self, config):
        self.config = config
        self.verbose = config.getoption("--cuda-health-verbose")
        self.sync_after_test = config.getoption("--cuda-health-sync")
        self.gc_after_test = config.getoption("--cuda-health-gc")

        self.last_healthy_test = None
        self.current_test = None
        self.test_count = 0
        self.health_failures = []
        self.cuda_available = False
        self.log_file = None
        self.worker_id = "main"

        # Detect xdist worker
        worker_id = os.environ.get("PYTEST_XDIST_WORKER")
        if worker_id:
            self.worker_id = worker_id
            # Each worker writes to its own log file
            self.log_file = open(f"cuda_health_{worker_id}.log", "w")

        # Try to import CUDA libraries
        try:
            import cupy as cp

            self.cp = cp
            self.cuda_available = True
        except ImportError:
            self.cp = None

        self._log(
            f"Plugin initialized: verbose={self.verbose}, "
            f"sync={self.sync_after_test}, cuda={self.cuda_available}, "
            f"worker={self.worker_id}",
            force=True,
        )

    def _log(self, msg, force=False):
        """Log a message if verbose mode is enabled."""
        if self.verbose or force:
            full_msg = f"[CUDA-HEALTH:{self.worker_id}] {msg}"
            # Always write to log file if we have one (xdist worker)
            if self.log_file:
                self.log_file.write(full_msg + "\n")
                self.log_file.flush()
            # Write to stderr (more likely to show with xdist)
            sys.stderr.write(full_msg + "\n")
            sys.stderr.flush()

    def _get_memory_info(self):
        """Get current CUDA memory info."""
        if not self.cuda_available:
            return None

        try:
            mempool = self.cp.get_default_memory_pool()

            info = {
                "device_used": mempool.used_bytes(),
                "device_total": mempool.total_bytes(),
            }

            # Pinned memory pool has different API
            try:
                pinned_mempool = self.cp.get_default_pinned_memory_pool()
                info["pinned_used"] = pinned_mempool.n_free_blocks()
            except Exception:
                pass

            return info
        except Exception as e:
            return {"error": str(e)}

    def _check_cuda_health(self):
        """Check if CUDA is in a healthy state."""
        if not self.cuda_available:
            return True, "CUDA not available"

        try:
            # Try a simple CUDA operation
            a = self.cp.array([1, 2, 3], dtype=self.cp.float32)
            b = a + 1
            self.cp.cuda.Stream.null.synchronize()
            del a, b
            return True, "OK"
        except Exception as e:
            error_msg = str(e)
            # Check for specific CUDA errors
            if "cudaErrorIllegalAddress" in error_msg:
                return False, f"ILLEGAL MEMORY ACCESS: {error_msg}"
            elif "cudaError" in error_msg:
                return False, f"CUDA ERROR: {error_msg}"
            else:
                return False, f"UNKNOWN ERROR: {error_msg}"

    def _sync_cuda(self):
        """Synchronize CUDA stream."""
        if not self.cuda_available:
            return

        try:
            self.cp.cuda.Stream.null.synchronize()
        except Exception as e:
            self._log(f"Sync failed: {e}", force=True)

    def _force_gc(self):
        """Force garbage collection and CUDA memory cleanup."""
        import gc

        gc.collect()

        if self.cuda_available:
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
                self.cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        """Called before each test runs."""
        self.current_test = item.nodeid
        self.test_count += 1

        if self.verbose:
            mem_info = self._get_memory_info()
            self._log(
                f"[{self.test_count}] SETUP: {item.nodeid} | "
                f"Memory: {mem_info}"
            )

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        """Called after each test runs."""
        # Force sync if requested
        if self.sync_after_test:
            self._sync_cuda()

        # Force GC if requested
        if self.gc_after_test:
            self._force_gc()

        # Check CUDA health
        healthy, msg = self._check_cuda_health()

        if not healthy:
            failure_info = {
                "test": item.nodeid,
                "test_number": self.test_count,
                "error": msg,
                "last_healthy_test": self.last_healthy_test,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "worker": self.worker_id,
            }
            self.health_failures.append(failure_info)

            # Build failure message
            failure_msg = (
                "\n"
                + "=" * 80
                + "\n"
                f"[CUDA-HEALTH:{self.worker_id}] !!! CUDA HEALTH CHECK FAILED !!!\n"
                f"  Current test: {item.nodeid}\n"
                f"  Test number: {self.test_count}\n"
                f"  Error: {msg}\n"
                f"  Last healthy test: {self.last_healthy_test}\n"
                + "=" * 80
                + "\n"
            )

            # Write to log file if xdist worker
            if self.log_file:
                self.log_file.write(failure_msg)
                self.log_file.flush()

            # Write to stderr (works better with xdist)
            sys.stderr.write(failure_msg)
            sys.stderr.flush()
        else:
            self.last_healthy_test = item.nodeid

            if self.verbose:
                mem_info = self._get_memory_info()
                self._log(
                    f"[{self.test_count}] TEARDOWN: {item.nodeid} | "
                    f"Health: {msg} | Memory: {mem_info}"
                )

    def pytest_sessionfinish(self, session, exitstatus):
        """Called at the end of the test session."""
        summary_lines = []

        if self.health_failures:
            summary_lines.append("\n" + "=" * 80)
            summary_lines.append(
                f"[CUDA-HEALTH:{self.worker_id}] SESSION SUMMARY - "
                "HEALTH CHECK FAILURES"
            )
            summary_lines.append("=" * 80)

            for failure in self.health_failures:
                summary_lines.append(
                    f"\nTest #{failure['test_number']}: {failure['test']}"
                )
                summary_lines.append(f"  Error: {failure['error']}")
                summary_lines.append(
                    f"  Last healthy test: {failure['last_healthy_test']}"
                )
                summary_lines.append(f"  Time: {failure['timestamp']}")

            summary_lines.append("\n" + "=" * 80)
            summary_lines.append(
                f"Total CUDA health failures: {len(self.health_failures)}"
            )
            summary_lines.append("=" * 80 + "\n")

            summary = "\n".join(summary_lines)
            sys.stderr.write(summary + "\n")
            sys.stderr.flush()
            if self.log_file:
                self.log_file.write(summary)

        elif self.test_count > 0:
            self._log(
                f"Session complete. All {self.test_count} tests "
                "passed CUDA health checks.",
                force=True,
            )

        # Close log file if we have one
        if self.log_file:
            self.log_file.close()


def pytest_configure(config):
    """Register the plugin if --cuda-health-check is specified."""
    if config.getoption("--cuda-health-check", default=False):
        plugin_name = "cuda_health_plugin_instance"
        if not config.pluginmanager.has_plugin(plugin_name):
            config.pluginmanager.register(
                CUDAHealthPlugin(config), plugin_name
            )

