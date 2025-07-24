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
import code
import contextlib
import os
import runpy
import sys

from cuml.accel import profilers


@contextlib.contextmanager
def _maybe_profile(
    path=None, source=None, namespace=None, profile=False, line_profile=False
):
    """A helper for managing optional application of the profilers."""
    with contextlib.ExitStack() as stack:
        if profile:
            stack.enter_context(profilers.profile())
            if line_profile:
                # If running both profilers, add a blank line between the reports
                stack.callback(print)
        if line_profile:
            stack.enter_context(
                profilers.LineProfiler(
                    path=path, source=source, namespace=namespace
                )
            )
        yield


def run_path(
    path: str, profile: bool = False, line_profile: bool = False
) -> None:
    """Run a script at a given path the same as ``python script``."""
    path = os.path.abspath(path)
    with _maybe_profile(path=path, profile=profile, line_profile=line_profile):
        runpy.run_path(path, run_name="__main__")


def run_module(module: str, profile: bool = False) -> None:
    """Run a module the same as ``python -m module``."""
    with _maybe_profile(profile=profile):
        runpy.run_module(module, run_name="__main__", alter_sys=True)


def run_interpreter(profile: bool = False) -> None:
    """Start an interactive interpreter the same as ``python``."""
    with _maybe_profile(profile=profile):
        if sys.flags.quiet:
            banner = ""
        else:
            banner = f"Python {sys.version} on {sys.platform}"
            if not sys.flags.no_site:
                cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                banner += "\n" + cprt
        code.interact(banner=banner, exitmsg="")


def run_source(
    source: str,
    filename: str = "<stdin>",
    profile: bool = False,
    line_profile: bool = False,
) -> None:
    """Run the given source, handling errors and exit the same as ``python -c source``."""
    namespace = {
        "__name__": "__main__",
        "__file__": filename,
        "__doc__": None,
        "__package__": None,
        "__spec__": None,
        "__loader__": None,
        "__cached__": None,
    }
    try:
        exec_source(
            source,
            namespace,
            filename=filename,
            profile=profile,
            line_profile=line_profile,
        )
    except SystemExit:
        raise
    except BaseException:
        typ, value, tb = sys.exc_info()
        # Drop our frames from the traceback before formatting
        sys.__excepthook__(
            typ,
            value.with_traceback(tb.tb_next.tb_next),
            tb.tb_next.tb_next,
        )
        # Exit on error with the proper code
        code = 130 if typ is KeyboardInterrupt else 1
        sys.exit(code)


def exec_source(
    source: str,
    namespace: dict,
    filename: str = "<stdin>",
    profile: bool = False,
    line_profile: bool = False,
) -> None:
    """Execute the given source in the context of a specified namespace.

    This differs from ``run_source`` in two ways:

    - It doesn't prepopulate the executing namespace
    - It doesn't cleanup tracebacks nor exit on failure
    """
    code = compile(source, filename, "exec")

    with _maybe_profile(
        source=source,
        namespace=namespace,
        profile=profile,
        line_profile=line_profile,
    ):
        exec(code, namespace)
