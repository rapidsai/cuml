#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.

"""Parse pytest XML results and analyze test failures.

This script parses pytest XML results and provides various ways to analyze and display
test failures, including grouping by error type, filtering, and different output formats.

Usage with pytest's @file feature:
--------------------------------
The script can output test names in a format compatible with pytest's @file feature,
which allows reading test names from a file. To use this:

1. Generate a file containing test names:
   $ python parse-pytest-results.py results.xml --format pytest > pytest.txt

2. Run the tests using pytest's @file syntax (requires pytest 8.2+):
   $ pytest -p cuml.accel --pyargs @pytest.txt

This approach is particularly useful when you want to:
- Re-run only the failed tests
- Run a specific group of failures (using --select-group)
- Run tests matching a pattern (using -k)
- Run the top N failing test groups (using -g N)

The @file feature automatically handles proper escaping of test names containing
special characters, making it more reliable than shell-based approaches.
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd


class Verbosity(IntEnum):
    """Verbosity levels for output."""

    BRIEF = 0  # Only show counts
    NORMAL = 1  # Show test names
    DETAILED = 2  # Show full details


@dataclass
class TestFailure:
    """Represents a failed test case with its details."""

    classname: str
    name: str
    time: float
    message: str
    details: Optional[str] = None

    def get_error_signature(self) -> str:
        """Extract a meaningful error signature for grouping.

        This method attempts to create a more useful grouping key by:
        1. Looking at both message and details
        2. Including assertion details for AssertionErrors
        3. Extracting meaningful parts of the error
        """
        # Start with the first line of the message
        lines = self.message.split("\n")
        if not lines:
            return "Unknown Error"

        error_type = lines[0]

        # Special handling for common error types
        if error_type.startswith("AssertionError"):
            # For assertion errors, include the actual assertion line
            if self.details:
                # Look for the assertion line in the details
                for line in self.details.split("\n"):
                    if line.strip().startswith(">") and "assert" in line:
                        # Extract the assertion line without the '>' prefix
                        assertion = line.strip()[1:].strip()
                        error_type = f"{error_type}: {assertion}"
                        break
            # If we have details with array comparisons, include that
            if self.details and "Arrays are not equal" in self.details:
                for line in self.details.split("\n"):
                    if "Mismatched elements:" in line:
                        error_type += f" ({line.strip()})"
                        break

        elif error_type.startswith("ValueError"):
            # For ValueErrors, include the specific error message
            if len(lines) > 1:
                error_type = f"{lines[0]}: {lines[1].strip()}"
            # Include any type information from the details
            if self.details:
                for line in self.details.split("\n"):
                    if "dtype" in line or "type" in line:
                        error_type += f" ({line.strip()})"
                        break

        elif error_type.startswith("TypeError"):
            # For TypeErrors, include the type mismatch information
            if self.details:
                for line in self.details.split("\n"):
                    if "expected" in line.lower() or "got" in line.lower():
                        error_type += f" ({line.strip()})"
                        break

        return error_type

    def __str__(self) -> str:
        """Format the test failure for display."""
        return (
            f"{self.classname}.{self.name} "
            f"({self.time:.3f}s)\n"
            f"Error: {self.message}\n"
            f"{self.details if self.details else ''}"
        )


class PytestResults:
    """Parser and container for pytest results."""

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.testsuite = self.root.find("testsuite")

    def get_summary(self) -> dict:
        """Get summary of test results."""
        return {
            "total": int(self.testsuite.get("tests", 0)),
            "failures": int(self.testsuite.get("failures", 0)),
            "errors": int(self.testsuite.get("errors", 0)),
            "skipped": int(self.testsuite.get("skipped", 0)),
            "time": float(self.testsuite.get("time", 0)),
        }

    def get_failed_tests(self) -> List[TestFailure]:
        """Get list of failed tests with details."""
        failures = []
        for testcase in self.testsuite.findall(".//testcase"):
            failure = testcase.find("failure")
            if failure is not None:
                failures.append(
                    TestFailure(
                        classname=testcase.get("classname", ""),
                        name=testcase.get("name", ""),
                        time=float(testcase.get("time", 0)),
                        message=failure.get("message", ""),
                        details=failure.text,
                    )
                )
        return failures

    def get_grouped_failures(self) -> Dict[str, List[TestFailure]]:
        """Group test failures by error signature."""
        grouped = defaultdict(list)
        for failure in self.get_failed_tests():
            # Use the error signature as the key
            key = failure.get_error_signature()
            grouped[key].append(failure)

        # Sort by number of occurrences
        return dict(
            sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert test failures to a pandas DataFrame."""
        failures = self.get_failed_tests()
        return pd.DataFrame([asdict(f) for f in failures])


def group_failures_df(df: pd.DataFrame) -> pd.DataFrame:
    """Group failures in DataFrame format and return summary statistics."""
    # Extract first line of error message for grouping
    df["error_type"] = df["message"].str.split("\n").str[0]

    # Group by error type and get counts
    grouped = (
        df.groupby("error_type")
        .agg({"classname": "count", "time": ["mean", "sum"]})
        .round(3)
    )

    # Flatten column names and rename
    grouped.columns = ["count", "avg_time", "total_time"]
    return grouped.sort_values("count", ascending=False)


def print_summary(results: PytestResults):
    """Print summary of test results."""
    summary = results.get_summary()
    click.echo("\nTest Summary:")
    click.echo(f"Total tests: {summary['total']}")
    click.echo(f"Failures: {summary['failures']}")
    click.echo(f"Errors: {summary['errors']}")
    click.echo(f"Skipped: {summary['skipped']}")
    click.echo(f"Time: {summary['time']:.2f}s\n")


def print_failures(failures: List[TestFailure], verbosity: Verbosity):
    """Print failed test cases."""
    if not failures:
        click.echo("No test failures found.")
        return

    click.echo(f"Found {len(failures)} test failures:\n")
    for i, failure in enumerate(failures, 1):
        if verbosity >= Verbosity.NORMAL:
            click.echo(f"[{i}] {failure.classname}.{failure.name}")
            click.echo(f"    Error: {failure.message}")
            if verbosity >= Verbosity.DETAILED and failure.details:
                click.echo("    Details:")
                for line in failure.details.splitlines():
                    click.echo(f"    {line}")
            click.echo()


def print_grouped_failures(
    grouped_failures: Dict[str, List[TestFailure]],
    verbosity: Verbosity,
    max_groups: int = -1,
    max_cases: int = 5,
):
    """Print failures grouped by error message.

    Args:
        grouped_failures: Dictionary of error messages to failure lists
        verbosity: Output verbosity level
        max_groups: Maximum number of groups to show (-1 for all)
        max_cases: Maximum number of cases to show per group (5 by default)
    """
    if not grouped_failures:
        click.echo("No test failures found.")
        return

    total = sum(len(failures) for failures in grouped_failures.values())
    total_groups = len(grouped_failures)
    shown_groups = list(grouped_failures.items())

    # If max_groups is a percentage (e.g., "80%"), convert to actual number
    if isinstance(max_groups, str) and max_groups.endswith("%"):
        percentage = float(max_groups[:-1]) / 100.0
        cumulative_failures = 0
        for i, (_, failures) in enumerate(shown_groups):
            cumulative_failures += len(failures)
            if cumulative_failures / total > percentage:
                max_groups = i + 1
                break

    if max_groups >= 0:
        shown_groups = shown_groups[:max_groups]

    click.echo(f"Found {total} test failures in {total_groups} groups:\n")

    for i, (error_msg, failures) in enumerate(shown_groups, 1):
        percentage = (len(failures) / total) * 100
        click.echo(f"Group {i}: {error_msg}")
        click.echo(f"Found {len(failures)} occurrences ({percentage:.1f}%)")

        if verbosity >= Verbosity.NORMAL:
            shown_cases = failures if max_cases < 0 else failures[:max_cases]
            for failure in shown_cases:
                click.echo(f"    {failure.classname}.{failure.name}")
                if verbosity >= Verbosity.DETAILED and failure.details:
                    click.echo("    Details:")
                    for line in failure.details.splitlines():
                        click.echo(f"        {line}")

            if max_cases >= 0 and len(failures) > max_cases:
                click.echo(f"    ... and {len(failures) - max_cases} more")
            click.echo()

    if max_groups >= 0 and total_groups > max_groups:
        click.echo(f"\n... and {total_groups - max_groups} more groups")


def print_summary_markdown(results: PytestResults):
    """Print summary of test results in markdown format."""
    summary = results.get_summary()
    click.echo("\n## Test Summary")
    click.echo("|Metric|Value|")
    click.echo("|------|-----|")
    click.echo(f"|Total tests|{summary['total']}|")
    click.echo(f"|Failures|{summary['failures']}|")
    click.echo(f"|Errors|{summary['errors']}|")
    click.echo(f"|Skipped|{summary['skipped']}|")
    click.echo(f"|Time|{summary['time']:.2f}s|\n")


def print_failures_markdown(failures: List[TestFailure], verbosity: Verbosity):
    """Print failed test cases in markdown format."""
    if not failures:
        click.echo("No test failures found.")
        return

    click.echo(f"\n## Test Failures ({len(failures)} total)\n")
    for i, failure in enumerate(failures, 1):
        if verbosity >= Verbosity.NORMAL:
            click.echo(f"### {i}. {failure.classname}.{failure.name}\n")
            click.echo("```")
            click.echo(f"Error: {failure.message}")
            if verbosity >= Verbosity.DETAILED and failure.details:
                click.echo("\nDetails:")
                click.echo(failure.details)
            click.echo("```\n")


def print_grouped_failures_markdown(
    grouped_failures: Dict[str, List[TestFailure]],
    verbosity: Verbosity,
    max_groups: int = -1,
    max_cases: int = 5,
):
    """Print failures grouped by error message in markdown format."""
    if not grouped_failures:
        click.echo("No test failures found.")
        return

    total = sum(len(failures) for failures in grouped_failures.values())
    total_groups = len(grouped_failures)
    shown_groups = list(grouped_failures.items())

    # Handle percentage-based max_groups
    if isinstance(max_groups, str) and max_groups.endswith("%"):
        percentage = float(max_groups[:-1]) / 100.0
        cumulative_failures = 0
        for i, (_, failures) in enumerate(shown_groups):
            cumulative_failures += len(failures)
            if cumulative_failures / total > percentage:
                max_groups = i + 1
                break

    if max_groups >= 0:
        shown_groups = shown_groups[:max_groups]

    click.echo(
        f"\n## Test Failures ({total} total in {total_groups} groups)\n"
    )

    for i, (error_msg, failures) in enumerate(shown_groups, 1):
        percentage = (len(failures) / total) * 100
        click.echo(f"### Group {i}: {error_msg}")
        click.echo(f"**{len(failures)} occurrences ({percentage:.1f}%)**\n")

        if verbosity >= Verbosity.NORMAL:
            shown_cases = failures if max_cases < 0 else failures[:max_cases]
            if shown_cases:
                click.echo("```")
                for failure in shown_cases:
                    click.echo(f"{failure.classname}.{failure.name}")
                    if verbosity >= Verbosity.DETAILED and failure.details:
                        click.echo("Details:")
                        click.echo(failure.details)
                        click.echo()
                click.echo("```")

            if max_cases >= 0 and len(failures) > max_cases:
                click.echo(
                    f"\n*... and {len(failures) - max_cases} more cases*\n"
                )

    if max_groups >= 0 and total_groups > max_groups:
        click.echo(f"\n*... and {total_groups - max_groups} more groups*")


def format_json_output(
    results: PytestResults,
    grouped_failures: Optional[Dict[str, List[TestFailure]]] = None,
    failures: Optional[List[TestFailure]] = None,
    max_groups: int = -1,
    max_cases: int = 5,
) -> dict:
    """Format test results as JSON."""
    summary = results.get_summary()
    output = {"summary": summary, "format_version": "1.0"}

    if grouped_failures is not None:
        total = sum(len(f) for f in grouped_failures.values())
        groups_data = []
        shown_groups = list(grouped_failures.items())

        # Handle percentage-based max_groups
        if isinstance(max_groups, str) and max_groups.endswith("%"):
            percentage = float(max_groups[:-1]) / 100.0
            cumulative_failures = 0
            for i, (_, failures) in enumerate(shown_groups):
                cumulative_failures += len(failures)
                if cumulative_failures / total > percentage:
                    max_groups = i + 1
                    break

        if max_groups >= 0:
            shown_groups = shown_groups[:max_groups]

        for error_msg, group_failures in shown_groups:
            group_data = {
                "error_message": error_msg,
                "count": len(group_failures),
                "percentage": (len(group_failures) / total) * 100,
                "failures": [],
            }

            shown_cases = (
                group_failures if max_cases < 0 else group_failures[:max_cases]
            )
            for failure in shown_cases:
                group_data["failures"].append(
                    {
                        "classname": failure.classname,
                        "name": failure.name,
                        "time": failure.time,
                        "message": failure.message,
                        "details": failure.details,
                    }
                )

            if max_cases >= 0 and len(group_failures) > max_cases:
                group_data["additional_cases"] = (
                    len(group_failures) - max_cases
                )

            groups_data.append(group_data)

        output["grouped_failures"] = {
            "total_failures": total,
            "total_groups": len(grouped_failures),
            "shown_groups": len(shown_groups),
            "groups": groups_data,
        }

    elif failures is not None:
        failures_data = []
        for failure in failures:
            failures_data.append(
                {
                    "classname": failure.classname,
                    "name": failure.name,
                    "time": failure.time,
                    "message": failure.message,
                    "details": failure.details,
                }
            )
        output["failures"] = failures_data

    return output


def format_pytest_output(
    results: PytestResults,
    grouped_failures: Optional[Dict[str, List[TestFailure]]] = None,
    failures: Optional[List[TestFailure]] = None,
    max_groups: int = -1,
    max_cases: int = 5,
) -> str:
    """Format test results as a list of test paths.

    Returns a string containing test paths separated by null characters that can be passed directly to pytest.
    Each test path is prefixed with "sklearn." to match the test organization.
    """

    def format_test_path(classname: str, name: str) -> str:
        """Format a single test path."""
        # Remove any existing sklearn prefix to avoid duplication
        if classname.startswith("sklearn."):
            classname = classname[8:]  # Remove "sklearn." prefix
        return f"sklearn.{classname}::{name}"

    if grouped_failures is not None:
        # For grouped failures, we'll use the first failure in each group
        shown_groups = list(grouped_failures.items())
        if max_groups >= 0:
            shown_groups = shown_groups[:max_groups]

        test_cases = []
        for _, group_failures in shown_groups:
            shown_cases = (
                group_failures if max_cases < 0 else group_failures[:max_cases]
            )
            for failure in shown_cases:
                test_cases.append(
                    format_test_path(failure.classname, failure.name)
                )
    elif failures is not None:
        test_cases = [format_test_path(f.classname, f.name) for f in failures]
    else:
        return ""

    # Return test paths separated by null characters
    return "\n".join(test_cases)


@click.command()
@click.argument("xml_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["brief", "normal", "detailed"]),
    default="normal",
    help="Output verbosity level",
)
@click.option(
    "--summary/--no-summary", default=True, help="Show/hide test summary"
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Write output to file instead of stdout",
)
@click.option(
    "-g",
    "--group",
    type=str,
    default=None,
    help="Group failures and show top N groups or N%",
)
@click.option(
    "--show-max",
    type=int,
    default=5,
    help="Maximum number of cases to show per group",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "markdown", "json", "pytest"]),
    default="console",
    help="Output format (console, markdown, json, or pytest)",
)
@click.option(
    "-k",
    "filter_expr",
    type=str,
    help="Filter test failures by expression (e.g. 'pca' to show only PCA-related failures)",
)
@click.option(
    "--select-group",
    type=int,
    help="Select a specific failure group by number to display",
)
def main(
    xml_file: Path,
    verbosity: str,
    summary: bool,
    output: Optional[Path],
    group: Optional[str],
    show_max: int,
    output_format: str,
    filter_expr: Optional[str],
    select_group: Optional[int],
):
    """Parse pytest XML results and display failure information.

    XML_FILE: Path to the pytest XML results file
    """
    try:
        results = PytestResults(xml_file)
        verb_level = Verbosity[verbosity.upper()]

        # Redirect output if specified
        if output:
            sys.stdout = open(output, "w")

        # Filter failures based on expression if provided
        def filter_failures(failures: List[TestFailure]) -> List[TestFailure]:
            if not filter_expr:
                return failures
            expr = filter_expr.lower()
            return [
                f
                for f in failures
                if expr in f.classname.lower() or expr in f.name.lower()
            ]

        if output_format == "json":
            # For JSON, we collect all data and output at once
            grouped = (
                results.get_grouped_failures() if group is not None else None
            )
            failures = results.get_failed_tests() if group is None else None

            # Apply filtering
            if failures is not None:
                failures = filter_failures(failures)
            if grouped is not None:
                # Rebuild grouped failures after filtering
                filtered_failures = filter_failures(results.get_failed_tests())
                grouped = defaultdict(list)
                for failure in filtered_failures:
                    key = failure.get_error_signature()
                    grouped[key].append(failure)
                grouped = dict(
                    sorted(
                        grouped.items(), key=lambda x: len(x[1]), reverse=True
                    )
                )

                # Handle group selection
                if select_group is not None:
                    if select_group < 1 or select_group > len(grouped):
                        click.echo(
                            f"Error: Group {select_group} not found", err=True
                        )
                        sys.exit(1)
                    # Get the selected group
                    selected_key = list(grouped.keys())[select_group - 1]
                    failures = grouped[selected_key]
                    grouped = None

            max_groups = (
                group
                if group and group.endswith("%")
                else int(group)
                if group
                else -1
            )

            output_data = format_json_output(
                results, grouped, failures, max_groups, show_max
            )
            click.echo(json.dumps(output_data, indent=2))

        elif output_format == "pytest":
            # For pytest format, we'll generate a pytest expression
            grouped = (
                results.get_grouped_failures() if group is not None else None
            )
            failures = results.get_failed_tests() if group is None else None

            # Apply filtering
            if failures is not None:
                failures = filter_failures(failures)
            if grouped is not None:
                # Rebuild grouped failures after filtering
                filtered_failures = filter_failures(results.get_failed_tests())
                grouped = defaultdict(list)
                for failure in filtered_failures:
                    key = failure.get_error_signature()
                    grouped[key].append(failure)
                grouped = dict(
                    sorted(
                        grouped.items(), key=lambda x: len(x[1]), reverse=True
                    )
                )

                # Handle group selection
                if select_group is not None:
                    if select_group < 1 or select_group > len(grouped):
                        click.echo(
                            f"Error: Group {select_group} not found", err=True
                        )
                        sys.exit(1)
                    # Get the selected group
                    selected_key = list(grouped.keys())[select_group - 1]
                    failures = grouped[selected_key]
                    grouped = None

            max_groups = (
                group
                if group and group.endswith("%")
                else int(group)
                if group
                else -1
            )

            pytest_expr = format_pytest_output(
                results, grouped, failures, max_groups, show_max
            )
            click.echo(pytest_expr)

        else:
            # For console and markdown formats
            if summary:
                if output_format == "markdown":
                    print_summary_markdown(results)
                else:
                    print_summary(results)

            if group is not None:
                grouped = results.get_grouped_failures()
                # Apply filtering to grouped failures
                if filter_expr:
                    filtered_failures = filter_failures(
                        results.get_failed_tests()
                    )
                    grouped = defaultdict(list)
                    for failure in filtered_failures:
                        key = failure.get_error_signature()
                        grouped[key].append(failure)
                    grouped = dict(
                        sorted(
                            grouped.items(),
                            key=lambda x: len(x[1]),
                            reverse=True,
                        )
                    )

                # Handle group selection
                if select_group is not None:
                    if select_group < 1 or select_group > len(grouped):
                        click.echo(
                            f"Error: Group {select_group} not found", err=True
                        )
                        sys.exit(1)
                    # Get the selected group
                    selected_key = list(grouped.keys())[select_group - 1]
                    failures = grouped[selected_key]
                    if output_format == "markdown":
                        print_failures_markdown(failures, verb_level)
                    else:
                        print_failures(failures, verb_level)
                else:
                    max_groups = group if group.endswith("%") else int(group)
                    if output_format == "markdown":
                        print_grouped_failures_markdown(
                            grouped, verb_level, max_groups, show_max
                        )
                    else:
                        print_grouped_failures(
                            grouped, verb_level, max_groups, show_max
                        )
            else:
                failures = results.get_failed_tests()
                # Apply filtering to failures
                failures = filter_failures(failures)
                if output_format == "markdown":
                    print_failures_markdown(failures, verb_level)
                else:
                    print_failures(failures, verb_level)

    except ET.ParseError as e:
        click.echo(f"Error parsing XML file: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
    finally:
        if output:
            sys.stdout.close()
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
