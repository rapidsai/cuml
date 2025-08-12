#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
from __future__ import annotations

import argparse
import sys
from textwrap import dedent

import cuml.accel.runners as runners
from cuml.accel.core import install


def error(msg: str, exit_code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse args for the `cuml.accel` CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m cuml.accel",
        description="Execute a script or module with `cuml.accel` enabled.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Examples:

            Execute a script with `cuml.accel` enabled:

              $ python -m cuml.accel myscript.py

            Trailing arguments are forwarded to the script:

              $ python -m cuml.accel myscript --some-option --another-option=example

            Instead of a script, a module may be specified instead.

              $ python -m cuml.accel -m mymodule --some-option

            If you also wish to use the `cudf.pandas` accelerator, you can invoke both
            as part of a single call like:

              $ python -m cudf.pandas -m cuml.accel myscript.py
            """
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase output verbosity (can be used multiple times, e.g. -vv). "
            "Default shows warnings only."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable the profiler.",
    )
    parser.add_argument(
        "--line-profile",
        action="store_true",
        help="Enable the line profiler.",
    )
    parser.add_argument(
        "--disable-uvm",
        action="store_true",
        help="Disable UVM (managed memory) allocations.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-m",
        dest="module",
        help="A module to execute",
    )
    group.add_argument(
        "-c",
        dest="cmd",
        help="Python source to execute, passed in as a string",
    )
    group.add_argument(
        "script",
        default="-",
        nargs="?",
        help="A script to execute",
    )
    parser.add_argument(
        "args",
        metavar="...",
        nargs=argparse.REMAINDER,
        help="Additional arguments to forward to script or module",
    )
    # We want to ignore all arguments after a module, cmd, or script are provided,
    # forwarding them on to the module/cmd/script. `script` is handled natively by
    # argparse, but `-m`/`-c` need some hacking to make work. We handle this by
    # splitting argv at the first of `-m`/`-c` provided (if any), parsing args up
    # to this point, then appending the remainder afterwards.
    m_index = c_index = len(argv)
    try:
        m_index = argv.index("-m")
    except ValueError:
        pass
    try:
        c_index = argv.index("-c")
    except ValueError:
        pass

    # Split at the first `-m foo`/`-c foo` found
    index = min(m_index, c_index)
    head = argv[: index + 2]
    tail = argv[index + 2 :]

    # Parse the head, then append the tail to `args`
    ns = parser.parse_args(head)
    ns.args.extend(tail)
    return ns


def main(argv: list[str] | None = None):
    """Run the `cuml.accel` CLI"""
    # Parse arguments
    ns = parse_args(sys.argv[1:] if argv is None else argv)

    # Parse verbose into log_level
    log_level = {0: "warn", 1: "info", 2: "debug"}.get(min(ns.verbose, 2))

    # Enable acceleration
    install(disable_uvm=ns.disable_uvm, log_level=log_level)

    if ns.module is not None:
        if ns.line_profile:
            error("--line-profile is not supported with -m")
        # Execute a module
        sys.argv[:] = [ns.module, *ns.args]
        runners.run_module(ns.module, profile=ns.profile)
    elif ns.cmd is not None:
        # Execute a cmd
        sys.argv[:] = ["-c", *ns.args]
        runners.run_source(
            ns.cmd, profile=ns.profile, line_profile=ns.line_profile
        )
    elif ns.script != "-":
        # Execute a script
        sys.argv[:] = [ns.script, *ns.args]
        runners.run_path(
            ns.script, profile=ns.profile, line_profile=ns.line_profile
        )
    else:
        sys.argv[:] = ["-", *ns.args]
        if sys.stdin.isatty():
            if ns.line_profile:
                error("--line-profile requires a script or cmd")
            # Start an interpreter
            runners.run_interpreter(profile=ns.profile)
        else:
            # Execute stdin
            runners.run_source(
                sys.stdin.read(),
                profile=ns.profile,
                line_profile=ns.line_profile,
            )


if __name__ == "__main__":
    main()
