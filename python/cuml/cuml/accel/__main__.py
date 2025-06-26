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
import code
import enum
import runpy
import sys
import warnings
from textwrap import dedent

from cuml.accel.core import install
from cuml.internals import logger


class ProfilingKind(enum.Enum):
    """Types of profiling output available."""

    NONE = "none"
    FORMATTED = "formatted"
    ANNOTATED = "annotated"
    RAW = "raw"


def setup_profiling(
    profiling_kind: ProfilingKind,
    script_path: str | None = None,
):
    """Set up console logging for profiling."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        raise RuntimeError(
            "Profiling dependencies not available. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
    else:
        # Import our profiling module for the formatted exporter
        from cuml.accel.profiling import (
            AnnotatedConsoleSpanExporter,
            RawSpanFormatter,
        )
        from cuml.accel.trace_formatter import AnnotatedTraceFormatter

        # Choose the appropriate exporter
        if profiling_kind == ProfilingKind.FORMATTED:
            console_exporter = ConsoleSpanExporter(
                formatter=AnnotatedTraceFormatter()
            )
        elif profiling_kind == ProfilingKind.ANNOTATED:
            console_exporter = AnnotatedConsoleSpanExporter(script_path)
        elif profiling_kind == ProfilingKind.RAW:
            console_exporter = ConsoleSpanExporter(
                formatter=RawSpanFormatter()
            )
        else:
            # This should not happen, but provide a fallback
            console_exporter = ConsoleSpanExporter()

        # Configure the tracer provider
        resource = Resource.create(
            {
                "service.name": "cuml-accel",
            }
        )
        tracer_provider = TracerProvider(resource=resource)

        # Add console exporter to output spans to stdout
        span_processor = BatchSpanProcessor(console_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Set the tracer provider as the global default
        trace.set_tracer_provider(tracer_provider)

        logger.info("Profiling enabled - spans will be output to console ")
        return True


def execute_source(source: str, filename: str = "<stdin>") -> None:
    """Execute source the same way python's interpreter would.

    source: str
        The source code to execute.
    filename: str, optional
        The filename to execute it as. Defaults to `"<stdin>"`
    """
    try:
        exec(compile(source, filename, "exec"))
    except SystemExit:
        raise
    except BaseException:
        typ, value, tb = sys.exc_info()
        # Drop our frame from the traceback before formatting
        sys.__excepthook__(
            typ,
            value.with_traceback(tb.tb_next),
            tb.tb_next,
        )
        # Exit on error with the proper code
        code = 130 if typ is KeyboardInterrupt else 1
        sys.exit(code)


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

            Enable profiling with:

              $ python -m cuml.accel --profile myscript.py

            This will instrument the code and track which operations are
            accelerated, and which operations fell back to CPU. After completion,
            a log of all operations will be printed to the console.

            Alternatively, you can enable annotated profiling with:

              $ python -m cuml.accel --profile-annotate myscript.py

            The annotated output shows the calling code with information
            on which lines had accelerated operations, and which lines
            had CPU fallback operations.

            For raw trace output (JSON format),  use:

              $ python -m cuml.accel --trace myscript.py

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
        "--disable-uvm",
        action="store_true",
        help="Disable UVM (managed memory) allocations.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling with formatted console output for tracing "
        "cuML acceleration operations.",
    )
    parser.add_argument(
        "--profile-annotate",
        action="store_true",
        help="Enable profiling with annotated console output showing calling code "
        "with acceleration status icons.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable profiling with raw console output for tracing "
        "cuML acceleration operations.",
    )
    # --convert-to-sklearn, --format, --output, and --cudf-pandas are deprecated
    # and hidden from the CLI --help with `argparse.SUPPRESS
    parser.add_argument("--convert-to-sklearn", help=argparse.SUPPRESS)
    parser.add_argument(
        "--format",
        choices=["pickle", "joblib"],
        type=str.lower,
        default="pickle",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        default="converted_sklearn_model.pkl",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cudf-pandas",
        action="store_true",
        help=argparse.SUPPRESS,
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

    # Set up profiling if requested
    if sum([ns.profile, ns.profile_annotate, ns.trace]) > 1:
        print("Error: Cannot use multiple profiling options simultaneously")
        print(
            "Use --profile for formatted output, --profile-annotate for annotated output, "
            "or --trace for raw output."
        )
        sys.exit(2)
    elif ns.profile or ns.profile_annotate or ns.trace:
        # Determine profiling kind
        if ns.profile:
            profiling_kind = ProfilingKind.FORMATTED
        elif ns.profile_annotate:
            profiling_kind = ProfilingKind.ANNOTATED
        else:  # ns.trace
            profiling_kind = ProfilingKind.RAW

        # Get script path for annotated output
        script_path = None
        if ns.profile_annotate and ns.script and ns.script != "-":
            script_path = ns.script

        try:
            setup_profiling(
                profiling_kind=profiling_kind,
                script_path=script_path,
            )
        except RuntimeError as e:
            print(f"Unable to enable profiling: {e}")
            sys.exit(1)

    # If the user requested a conversion, handle it and exit
    if ns.convert_to_sklearn:
        warnings.warn(
            "`--convert-to-sklearn`, `--format`, and `--output` are deprecated and will "
            "be removed in 25.10. Estimators created with `cuml.accel` may now be "
            "serialized and loaded in environments without `cuml` without the need for "
            "running a conversion step.",
            FutureWarning,
        )
        with open(ns.convert_to_sklearn, "rb") as f:
            if ns.format == "pickle":
                import pickle as serializer
            elif ns.format == "joblib":
                import joblib as serializer
            estimator = serializer.load(f)

        with open(ns.output, "wb") as f:
            serializer.dump(estimator, f)
        sys.exit()

    # Enable cudf.pandas if requested
    if ns.cudf_pandas:
        warnings.warn(
            "`--cudf-pandas` is deprecated and will be removed in 25.10. Instead, please "
            "invoke both accelerators explicitly like\n\n"
            "  $ python -m cudf.pandas -m cuml.accel ...",
            FutureWarning,
        )
        import cudf.pandas

        cudf.pandas.install()

    # Parse verbose into log_level
    default_logger_level_index = list(logger.level_enum).index(
        logger.level_enum.warn
    )
    log_level = list(logger.level_enum)[
        max(0, default_logger_level_index - ns.verbose)
    ]

    # Enable acceleration
    install(disable_uvm=ns.disable_uvm, log_level=log_level)

    if ns.module is not None:
        # Execute a module
        sys.argv[:] = [ns.module, *ns.args]
        runpy.run_module(ns.module, run_name="__main__", alter_sys=True)
    elif ns.cmd is not None:
        # Execute a cmd
        sys.argv[:] = ["-c", *ns.args]
        execute_source(ns.cmd, "<stdin>")
    elif ns.script != "-":
        # Execute a script
        sys.argv[:] = [ns.script, *ns.args]
        runpy.run_path(ns.script, run_name="__main__")
    else:
        sys.argv[:] = ["-", *ns.args]
        if sys.stdin.isatty():
            # Start an interpreter as similar to `python` as possible
            if sys.flags.quiet:
                banner = ""
            else:
                banner = f"Python {sys.version} on {sys.platform}"
                if not sys.flags.no_site:
                    cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                    banner += "\n" + cprt
            code.interact(banner=banner, exitmsg="")
        else:
            # Execute stdin
            execute_source(sys.stdin.read(), "<stdin>")


if __name__ == "__main__":
    main()
