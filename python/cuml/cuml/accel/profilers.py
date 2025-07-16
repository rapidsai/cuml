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
from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("profile", "ProfileResults", "MethodStats")


class Callback:
    """An abstract callback interface."""

    def _on_gpu_call(self, qualname: str, duration: float) -> None:
        """Called after a successful GPU method run.

        Parameters
        ----------
        qualname : str
            The qualified method or function name.
        duration : float
            The method runtime duration.
        """
        pass

    def _on_cpu_call(
        self, qualname: str, duration: float, reason: str | None = None
    ) -> None:
        """Called after a successful CPU method run.

        Parameters
        ----------
        qualname : str
            The qualified method or function name.
        duration : float
            The method runtime duration.
        reason : str, optional
            An optional reason for why a CPU instead of GPU method was called.
        """
        pass


# A list of active callbacks
_CALLBACKS: list[Callback] = []


def format_duration(duration: float) -> str:
    """Format a duration as a concise, human-readable string."""

    def format_float(x):
        return f"{x:.3f}".rstrip("0").rstrip(".")

    if 0 < duration < 0.001:
        return f"{format_float(duration * 1000)}ms"
    elif duration < 1:
        return f"{format_float(duration)}s"
    elif duration < 60:
        return f"{duration:.1f}s"
    elif duration < 3600:
        m = int(duration // 60)
        s = format_float(duration % 60)
        return f"{m}m{s}s"
    else:
        h = int(duration // 3600)
        r = duration % 3600
        m = int(r // 60)
        s = format_float(r % 60)
        return f"{h}h{m}m{s}s"


@contextmanager
def track_gpu_call(qualname: str) -> Iterator[None]:
    """A contextmanager for tracking a potential GPU method call in the profilers."""
    start = perf_counter()
    try:
        yield
    except UnsupportedOnGPU:
        raise
    finally:
        duration = perf_counter() - start
        for callback in _CALLBACKS:
            callback._on_gpu_call(qualname, duration)


@contextmanager
def track_cpu_call(qualname: str, reason: str | None = None) -> Iterator[None]:
    """A contextmanager for tracking a CPU method call in the profilers."""
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        for callback in _CALLBACKS:
            callback._on_cpu_call(qualname, duration, reason=reason)


@contextmanager
def profile(print_report: bool = True) -> Iterator[ProfileResults]:
    """Profile a section of code.

    This will collect stats on all accelerated (or potentially-accelerated)
    method and function calls within the context.

    Parameters
    ----------
    print_report : bool, optional
        Whether to print the report automatically upon exiting the context.
        Defaults to True.

    Returns
    -------
    results : ProfileResults
        A record of the profile results within the context.
    """
    results = ProfileResults()
    _CALLBACKS.append(results)
    try:
        yield results
    finally:
        _CALLBACKS.remove(results)
        if print_report:
            results.print_report()


class MethodStats:
    """Call statistics on method.

    Attributes
    ----------
    gpu_calls : int
        The number of calls that ran on GPU.
    gpu_time : float
        The cumulative runtime of GPU calls.
    cpu_calls : int
        The number of calls that ran on CPU.
    cpu_time : float
        The cumulative runtime of CPU calls.
    fallback_reasons : set[str]
        A collection of reasons for why CPU fallback methods were called.
    """

    def __init__(self):
        self.gpu_calls = 0
        self.gpu_time = 0.0
        self.cpu_calls = 0
        self.cpu_time = 0.0
        self.fallback_reasons = set()


class ProfileResults(Callback):
    """Results from `cuml.accel.profile`.

    Attributes
    ----------
    method_calls : dict[str, MethodStats]
        A mapping of qualified method name to statistics about calls to
        that method.
    """

    def __init__(self):
        self.method_calls = {}

    def _on_gpu_call(self, qualname: str, duration: float) -> None:
        if (stats := self.method_calls.get(qualname)) is None:
            stats = self.method_calls[qualname] = MethodStats()
        stats = self.method_calls[qualname]
        stats.gpu_calls += 1
        stats.gpu_time += duration

    def _on_cpu_call(
        self, qualname: str, duration: float, reason: str | None = None
    ) -> None:
        if (stats := self.method_calls.get(qualname)) is None:
            stats = self.method_calls[qualname] = MethodStats()
        stats.cpu_calls += 1
        stats.cpu_time += duration
        if reason is not None:
            stats.fallback_reasons.add(reason)

    def print_report(self) -> None:
        """Print a report of the results."""
        from rich.console import Console
        from rich.style import Style
        from rich.table import Table

        console = Console()

        table = Table(
            title="cuml.accel profile",
            title_justify="left",
            title_style=Style(bold=True),
            caption_style=Style(),
            caption_justify="left",
        )
        table.add_column("Function", no_wrap=True)
        table.add_column("GPU calls", justify="right", no_wrap=True)
        table.add_column("GPU time", justify="right", no_wrap=True)
        table.add_column("CPU calls", justify="right", no_wrap=True)
        table.add_column("CPU time", justify="right", no_wrap=True)

        fallbacks = []
        gpu_calls = cpu_calls = gpu_total_time = cpu_total_time = 0
        for function, stats in sorted(self.method_calls.items()):
            gpu_calls += stats.gpu_calls
            gpu_total_time += stats.gpu_time
            cpu_calls += stats.cpu_calls
            cpu_total_time += stats.cpu_time
            if stats.fallback_reasons:
                fallbacks.append((function, stats.fallback_reasons))

            table.add_row(
                function,
                str(stats.gpu_calls),
                format_duration(stats.gpu_time),
                str(stats.cpu_calls),
                format_duration(stats.cpu_time),
            )

        table.add_section()
        table.add_row(
            "Total",
            str(gpu_calls),
            format_duration(gpu_total_time),
            str(cpu_calls),
            format_duration(cpu_total_time),
        )

        console.print(table)

        if fallbacks:
            parts = [
                (
                    "Not all operations ran on the GPU. The following functions "
                    "required CPU fallback for the following reasons:\n"
                )
            ]
            for function, reasons in fallbacks:
                parts.append(f"* {function}")
                for reason in sorted(reasons):
                    parts.append(f"  - {reason}")
            console.print("\n".join(parts), highlight=False)


class LineStats:
    """Statistics about a single line in a script.

    Attributes
    ----------
    total_time : float
        The cumulative runtime spent on this line.
    count : int
        The number of times this line was hit during execution.
    gpu_time : float
        The cumulative amount of time GPU calls were made during this line.
    cpu_fallback : bool
        Whether a CPU fallback was called while on this line.
    """

    def __init__(self):
        self.total_time = 0
        self.count = 0
        self.gpu_time = 0
        self.cpu_fallback = False


class LineProfiler(Callback):
    """A line profiler, tracking line-level stats about a script.

    Not intended for direct use, use the CLI's ``--line-profile`` flag or the
    ``%%cuml.accel.line_profile`` IPython magic instead.

    Parameters
    ----------
    source : str, optional
        The script's source code. If not provided, will be inferred from `path`.
    path : str, optional
        A path to the script's source. May be omitted if the script's executing
        namespace is provided instead (useful if executing source code without a path).
    namespace : dict, optional
        The script's executing namespace. When executing source that lacks a unique path,
        the namespace is needed to temporarily inject a value so the profiler can know
        which frames to trace.
    """

    FLAG = "__cuml_accel_line_profiler_enabled__"

    def __init__(
        self,
        source: str | None = None,
        path: str | None = None,
        namespace: dict | None = None,
    ):
        if path is not None:
            self._should_trace = lambda frame: frame.f_code.co_filename == path
        else:
            self._should_trace = lambda frame: frame.f_globals.get(
                self.FLAG, False
            )
        if source is None:
            if path is None:
                raise ValueError("Must provide either `path` or `source`")
            with open(path, "r") as f:
                source = f.read()

        self.source = source
        self.path = path
        self.namespace = namespace

        self._total_time = 0.0
        self._total_gpu_time = 0.0
        self._stats = defaultdict(LineStats)
        self._timers = []
        self._new_frame = False

    def _on_gpu_call(self, qualname: str, duration: float) -> None:
        for lineno, _ in self._timers:
            self._stats[lineno].gpu_time += duration
        self._total_gpu_time += duration

    def _on_cpu_call(
        self, qualname: str, duration: float, reason: str | None = None
    ) -> None:
        for lineno, _ in self._timers:
            self._stats[lineno].cpu_fallback = True

    def __enter__(self) -> LineProfiler:
        self._old_trace = sys.gettrace()
        self._start_time = perf_counter()

        sys.settrace(self._trace)
        if self.namespace is not None:
            self.namespace[self.FLAG] = True

        _CALLBACKS.append(self)
        return self

    def __exit__(self, *args) -> None:
        _CALLBACKS.remove(self)
        self._total_time += perf_counter() - self._start_time

        if self.namespace is not None:
            self.namespace.pop(self.FLAG, None)
        sys.settrace(self._old_trace)

        self.print_report()

    def _maybe_pop_timer(self):
        if self._new_frame:
            self._new_frame = False
        else:
            lineno, start = self._timers.pop()
            duration = perf_counter() - start
            stats = self._stats[lineno]
            stats.count += 1
            stats.total_time += duration

    def _trace(self, frame, event, arg):
        if not self._should_trace(frame):
            # This frame is not directly in this script, no need to trace
            return None

        if event == "line":
            # New line, maybe pop the last line's timer and push a new one.
            self._maybe_pop_timer()
            self._timers.append((frame.f_lineno, perf_counter()))
        elif event == "return":
            # Returning from a frame, maybe pop the last line's timer.
            self._maybe_pop_timer()
        elif event == "call":
            # New function call, mark a new frame.
            self._new_frame = True

        return self._trace

    def print_report(self) -> None:
        """Print a report of the results."""
        from rich.console import Console
        from rich.style import Style
        from rich.syntax import Syntax
        from rich.table import Table

        gpu_percent = 100 * self._total_gpu_time / self._total_time

        table = Table(
            title="cuml.accel line profile",
            title_justify="left",
            title_style=Style(bold=True),
            caption=f"Ran in {format_duration(self._total_time)}, {gpu_percent:.1f}% on GPU",
            caption_style=Style(),
            caption_justify="left",
        )
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("N", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("GPU %", justify="right", no_wrap=True)
        table.add_column("Source", justify="left")

        for lineno, line in enumerate(self.source.splitlines(), 1):
            stats = self._stats[lineno]

            if stats.count == 0:
                row = ("", "", "")
            else:
                if not stats.gpu_time and not stats.cpu_fallback:
                    gpu_percent = "-"
                elif stats.total_time > 0:
                    gpu_percent = (
                        f"{100 * stats.gpu_time // stats.total_time:.1f}"
                    )
                else:
                    gpu_percent = "0.0"

                if stats.total_time < 0.001:
                    # Too short, not worth displaying
                    time = "-"
                else:
                    time = format_duration(stats.total_time)

                row = (str(stats.count), time, gpu_percent)

            table.add_row(
                str(lineno),
                *row,
                Syntax(line, "python", theme="ansi_light"),
            )
        Console().print(table)
