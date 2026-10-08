# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for suite-driven neutral-v2 benchmarks."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import re
import sys
from pathlib import Path

from ._logging import logger
from .backends import get_backend
from .suite import (
    BUILTIN_SUITES,
    SuiteError,
    load_suite_reference,
    suite_profile_names,
)

DEFAULT_SUITE = "cuml_sg"


def _parser(
    profile_names: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    """Build the benchmark argument parser with suite profile choices."""
    parser = argparse.ArgumentParser(prog="python -m cuml.benchmark")
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help=(
            "built-in suite name or strict YAML manifest path; built-ins: "
            + ", ".join(sorted(BUILTIN_SUITES))
            + f" (default: {DEFAULT_SUITE})"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "neutral-v2 JSON artifact path; defaults to a timestamped file "
            "in the current directory"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume --output, preserving successful cases and retrying failures",
    )
    profile_metavar = (
        "{" + ",".join(profile_names) + "}" if profile_names else "PROFILE"
    )
    parser.add_argument(
        "--profile",
        metavar=profile_metavar,
        help="profile defined by the selected suite (default: standard)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report every warmup and measurement repetition",
    )
    return parser


def _output_suite_name(suite) -> str:
    """Return a filename-safe suite name."""
    path = str(suite.path)
    name = (
        path.removeprefix("builtin:")
        if path.startswith("builtin:")
        else suite.name
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_").lower()
    return slug or "benchmark"


def default_output_path(suite, now: dt.datetime | None = None) -> Path:
    """Return an unused timestamped artifact path in the current directory.

    Parameters
    ----------
    suite : Suite
        Suite used to name the artifact.
    now : datetime, optional
        Timestamp to use; defaults to the current UTC time.
    """
    timestamp = (now or dt.datetime.now(dt.timezone.utc)).astimezone(
        dt.timezone.utc
    )
    stem = (
        f"{_output_suite_name(suite)}-{suite.profile_name}-"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )
    candidate = Path.cwd() / f"{stem}.json"
    suffix = 2
    while candidate.exists():
        candidate = Path.cwd() / f"{stem}-{suffix}.json"
        suffix += 1
    return candidate


def _suite_aware_parser(argv: list[str]) -> argparse.ArgumentParser:
    """Build an argument parser using the selected suite's profiles."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--suite", default=DEFAULT_SUITE)
    probed, _ = probe.parse_known_args(argv)
    try:
        return _parser(suite_profile_names(probed.suite))
    except SuiteError:
        # Full parsing and suite loading below will report the actionable
        # manifest error. Generic help should remain available meanwhile.
        pass
    return _parser()


@contextlib.contextmanager
def _console_logging(verbose: bool):
    """Configure only benchmark output, restoring caller settings afterward."""
    target = logging.getLogger("cuml.benchmark")
    handlers, level, propagate = (
        target.handlers,
        target.level,
        target.propagate,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    target.handlers = [handler]
    target.setLevel(logging.DEBUG if verbose else logging.INFO)
    target.propagate = False
    try:
        yield
    finally:
        target.handlers = handlers
        target.setLevel(level)
        target.propagate = propagate
        handler.close()


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark command and return its exit status.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to sys.argv[1:].
    """
    argument_values = sys.argv[1:] if argv is None else argv
    parser = _suite_aware_parser(argument_values)
    args = parser.parse_args(argument_values)
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    try:
        suite = load_suite_reference(args.suite, args.profile)
        backend = get_backend(suite.implementation)
        backend.bootstrap_process()
        backend.prepare_process()
        output = (
            Path(args.output).resolve()
            if args.output
            else default_output_path(suite)
        )
        with _console_logging(args.verbose):
            if args.output is None:
                logger.info("Writing benchmark artifact to %s", output)
            # Import only after backend startup and prerequisite validation.
            from .harness import run_suite

            artifact = run_suite(
                suite,
                output,
                [sys.executable, "-m", "cuml.benchmark", *argument_values],
                resume=args.resume,
            )
    except SuiteError as exc:
        parser.error(str(exc))
    return int(
        any(
            result["outcome"]["status"] == "failed"
            for result in artifact["results"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
