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
from functools import cache
from time import perf_counter
from typing import Iterator

from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("profile", "ProfileResults", "MethodStats")


@cache
def get_syntax_theme():
    """Get our custom syntax theme.

    Defined as a local cached function to avoid the `rich` import until needed."""
    from pygments.token import Token
    from rich.style import Style
    from rich.syntax import SyntaxTheme

    class Theme(SyntaxTheme):
        """A syntax theme approximating the default syntax used to highlight
        in Jupyter notebooks. This works well in both light and dark terminals,
        as well as in light & dark themed notebooks. Only ascii colors and simple
        styles are used, making this broadly applicable"""

        styles = {
            Token.Keyword.Namespace: Style(color="green", bold=True),
            Token.Operator.Word: Style(color="green", bold=True),
            Token.Keyword: Style(color="green", bold=True),
            Token.Comment: Style(dim=True),
            Token.Name.Builtin: Style(color="green"),
            Token.Keyword.Constant: Style(color="green"),
            Token.Literal.Number: Style(color="green"),
            Token.Literal.String: Style(color="red"),
        }

        @cache
        def get_style_for_token(self, token_type):
            # Tokens are hierarchical and singletons. We traverse upwards,
            # applying the first style found that matches and cache the result
            # to avoid doing this again.
            token = token_type
            while token:
                try:
                    return self.styles[token]
                except KeyError:
                    pass
                token = token[:-1]
            # Default
            return Style.null()

        def get_background_style(self):
            return Style.null()

    return Theme()


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
    """Format a duration as a concise, human-readable string.

    Examples
    --------
    >>> format_duration(0)
    '0s'
    >>> format_duration(0.1234)
    '123.4ms'
    >>> format_duration(65.2)
    '1m5.2s'
    >>> format_duration(3612.3)
    '1h0m12.3s'
    """
    h, rem = divmod(duration, 3600)
    m, frac = divmod(rem, 60)
    unit = "s"
    if not h and not m and frac:
        if frac < 0.001:
            frac *= 1_000_000
            unit = "µs"
        elif frac < 1:
            frac *= 1000
            unit = "ms"

    frac_str = f"{frac:.1f}".rstrip("0").rstrip(".")

    if h:
        return f"{int(h)}h{int(m)}m{frac_str}s"
    elif m:
        return f"{int(m)}m{frac_str}s"
    return f"{frac_str}{unit}"


@contextmanager
def track_gpu_call(qualname: str) -> Iterator[None]:
    """A contextmanager for tracking a potential GPU method call in the profilers."""
    supported = True
    start = perf_counter()
    try:
        yield
    except UnsupportedOnGPU:
        supported = False
        raise
    finally:
        if supported:
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
def profile(quiet: bool = False) -> Iterator[ProfileResults]:
    """Profile a section of code.

    This will collect stats on all accelerated (or potentially-accelerated)
    method and function calls within the context, and output a report summarizing
    what methods ``cuml.accel`` was able to accelerate, and what methods required
    a CPU fallback.

    ``cuml.accel.profile`` provides programmatic access to this profiler.
    Alternatively, you may use the ``--profile`` flag when running under the CLI,
    or the ``%cuml.accel.profile`` IPython magic when running in IPython or a
    notebook environment.

    Parameters
    ----------
    quiet : bool, optional
        Set to True to skip printing the report automatically upon exiting the context.

    Returns
    -------
    results : ProfileResults
        A record of the profile results within the context.

    Examples
    --------
    As part of ``cuml.accel``, the profiler only works if the accelerator is
    installed. You may accomplish this programmatically with `cuml.accel.install`,
    or through an alternative method like the CLI (``python -m cuml.accel``) or the
    IPython magic (``%load_ext cuml.accel``).

    >>> import cuml
    >>> cuml.accel.install()  # doctest: +SKIP

    Once The accelerator is active, you're free to start running some scikit-learn
    code. The profiler helps you understand when ``cuml.accel`` was able to accelerate
    your code, and when it needed to fallback to CPU.

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge

    To profile only certain sections of your code, wrap them in a ``profile``
    contextmanager.

    >>> with cuml.accel.profile():  # doctest: +SKIP
    ...     X, y = make_regression()
    ...     model = Ridge()
    ...     model.fit(X, y)
    ...     model.predict(X)
    ...
    cuml.accel profile
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃ Function      ┃ GPU calls ┃ GPU time ┃ CPU calls ┃ CPU time ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
    │ Ridge.fit     │         1 │    167ms │         0 │       0s │
    │ Ridge.predict │         1 │    1.2ms │         0 │       0s │
    ├───────────────┼───────────┼──────────┼───────────┼──────────┤
    │ Total         │         2 │  168.2ms │         0 │       0s │
    └───────────────┴───────────┴──────────┴───────────┴──────────┘
    """
    results = ProfileResults()
    _CALLBACKS.append(results)
    try:
        yield results
    finally:
        _CALLBACKS.remove(results)
        if not quiet:
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
            if stats.cpu_calls:
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
            table.caption = "\n".join(parts)

        Console().print(table)


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
        The script's source code. If not provided, will be inferred from `filename`.
    filename : str, optional
        The filename for the source being profiled. If not provided, the `start` method
        must be called at the top level in the executing source.
    quiet : bool, optional
        Set to True to skip printing the report automatically upon exiting the context.
    """

    def __init__(
        self,
        source: str | None = None,
        filename: str | None = None,
        quiet: bool = False,
    ):
        if source is None:
            if filename is None:
                raise ValueError("Must provide either `filename` or `source`")
            with open(filename, "r") as f:
                source = f.read()

        self.source = source
        self.filename = filename
        self.quiet = quiet

        self.total_time = 0.0
        self.total_gpu_time = 0.0
        self.line_stats = defaultdict(LineStats)
        self._timers = []
        self._new_frame = False
        self._offset = 0

    def _on_gpu_call(self, qualname: str, duration: float) -> None:
        for lineno, _ in self._timers:
            self.line_stats[lineno].gpu_time += duration
        self.total_gpu_time += duration

    def _on_cpu_call(
        self, qualname: str, duration: float, reason: str | None = None
    ) -> None:
        for lineno, _ in self._timers:
            self.line_stats[lineno].cpu_fallback = True

    def start(self):
        """Start the line profiler.

        In normal usage this is called automatically within `__enter__`.

        When running within a cell, a call to `start` is injected into the source
        and the profiler isn't started until that line is hit. This makes it easier
        to compose ``%%cuml.accel.line_profile`` with other cell magics.
        """
        if not hasattr(self, "_old_trace"):
            if self.filename is None:
                parent = sys._getframe().f_back
                self.filename = parent.f_code.co_filename
                self._offset = parent.f_lineno
                parent.f_trace = self._trace

            self._old_trace = sys.gettrace()
            self._start_time = perf_counter()

            sys.settrace(self._trace)
            _CALLBACKS.append(self)

    def __enter__(self):
        if self.filename is not None:
            self.start()
        return self

    def __exit__(self, *args) -> None:
        try:
            _CALLBACKS.remove(self)
        except ValueError:
            pass

        if hasattr(self, "_start_time"):
            self.total_time += perf_counter() - self._start_time
        if (old_trace := getattr(self, "_old_trace", None)) is not None:
            sys.settrace(old_trace)
        if not self.quiet:
            self.print_report()

    def _maybe_pop_timer(self):
        if self._new_frame:
            self._new_frame = False
        elif self._timers:
            lineno, start = self._timers.pop()
            duration = perf_counter() - start
            stats = self.line_stats[lineno]
            stats.count += 1
            stats.total_time += duration

    def _trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.filename:
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

        gpu_percent = 100 * self.total_gpu_time / self.total_time

        syntax_theme = get_syntax_theme()

        table = Table(
            title="cuml.accel line profile",
            title_justify="left",
            title_style=Style(bold=True),
            caption=f"Ran in {format_duration(self.total_time)}, {gpu_percent:.1f}% on GPU",
            caption_style=Style(),
            caption_justify="left",
        )
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("N", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("GPU %", justify="right", no_wrap=True)
        table.add_column("Source", justify="left")

        # We always display the time for GPU/CPU fallback lines. For non-accel-related
        # lines, we only display the time if it was "long enough". By default this is
        # 1 ms, except for very short runs.
        min_non_accel_time = min(self.total_time / 1000, 0.001)

        for lineno, line in enumerate(self.source.rstrip().splitlines(), 1):
            stats = self.line_stats[lineno + self._offset]

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

                if (
                    not stats.gpu_time
                    and not stats.cpu_fallback
                    and stats.total_time < min_non_accel_time
                ):
                    # Time is very short and not a method relevant to the accelerator,
                    # not worth displaying
                    time = "-"
                else:
                    time = format_duration(stats.total_time)

                row = (str(stats.count), time, gpu_percent)

            table.add_row(
                str(lineno),
                *row,
                Syntax(line, "python", theme=syntax_theme),
            )
        Console().print(table)
