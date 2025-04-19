#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.

"""Compare two pytest XML results files and show differences in test outcomes.

This script takes two pytest XML results files and analyzes the differences between them,
showing which tests:
- Failed in the second file but passed in the first (regressions)
- Passed in the second file but failed in the first (fixes)
- Were skipped in one file but not the other
- Were added or removed between the two files

Usage:
    $ python diff-pytest-results.py baseline.xml current.xml
"""

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

import click


class TestStatus(IntEnum):
    """Test status enumeration."""

    PASSED = 0
    FAILED = 1
    ERROR = 2
    SKIPPED = 3


@dataclass
class TestResult:
    """Represents a test case with its result."""

    classname: str
    name: str
    time: float
    status: TestStatus
    message: Optional[str] = None
    details: Optional[str] = None

    def get_full_name(self) -> str:
        """Get the full test name including class."""
        return f"{self.classname}.{self.name}"

    def __str__(self) -> str:
        """Format the test result for display."""
        status_str = self.status.name.lower()
        result = f"{self.get_full_name()} ({self.time:.3f}s) [{status_str}]"
        if self.message:
            result += f"\nError: {self.message}"
        if self.details:
            result += f"\n{self.details}"
        return result

    def to_dict(self) -> dict:
        """Convert test result to dictionary for JSON output."""
        return {
            "classname": self.classname,
            "name": self.name,
            "time": self.time,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
        }


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

    def get_test_results(self) -> Dict[str, TestResult]:
        """Get all test results indexed by full test name."""
        results = {}
        for testcase in self.testsuite.findall(".//testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time = float(testcase.get("time", 0))
            full_name = f"{classname}.{name}"

            # Check for failure
            failure = testcase.find("failure")
            if failure is not None:
                results[full_name] = TestResult(
                    classname=classname,
                    name=name,
                    time=time,
                    status=TestStatus.FAILED,
                    message=failure.get("message"),
                    details=failure.text,
                )
                continue

            # Check for error
            error = testcase.find("error")
            if error is not None:
                results[full_name] = TestResult(
                    classname=classname,
                    name=name,
                    time=time,
                    status=TestStatus.ERROR,
                    message=error.get("message"),
                    details=error.text,
                )
                continue

            # Check for skipped
            skipped = testcase.find("skipped")
            if skipped is not None:
                results[full_name] = TestResult(
                    classname=classname,
                    name=name,
                    time=time,
                    status=TestStatus.SKIPPED,
                    message=skipped.get("message"),
                )
                continue

            # Test passed
            results[full_name] = TestResult(
                classname=classname,
                name=name,
                time=time,
                status=TestStatus.PASSED,
            )

        return results


def analyze_differences(
    baseline: PytestResults, current: PytestResults
) -> Dict[str, List[TestResult]]:
    """Analyze differences between baseline and current test results.

    Returns a dictionary with the following keys:
    - regressions: Tests that passed in baseline but failed in current
    - fixes: Tests that failed in baseline but passed in current
    - skip_changes: Tests that were skipped in one file but not the other
    - added: Tests that only exist in current
    - removed: Tests that only exist in baseline
    """
    baseline_results = baseline.get_test_results()
    current_results = current.get_test_results()

    # Get sets of test names
    baseline_tests = set(baseline_results.keys())
    current_tests = set(current_results.keys())

    # Find added and removed tests
    added_tests = current_tests - baseline_tests
    removed_tests = baseline_tests - current_tests

    # Find tests that exist in both files
    common_tests = baseline_tests & current_tests

    # Analyze status changes
    regressions = []
    fixes = []
    skip_changes = []

    for test_name in common_tests:
        baseline_result = baseline_results[test_name]
        current_result = current_results[test_name]

        # Check for regressions (passed -> failed/error)
        if (
            baseline_result.status == TestStatus.PASSED
            and current_result.status in [TestStatus.FAILED, TestStatus.ERROR]
        ):
            regressions.append(current_result)

        # Check for fixes (failed/error -> passed)
        if (
            baseline_result.status in [TestStatus.FAILED, TestStatus.ERROR]
            and current_result.status == TestStatus.PASSED
        ):
            fixes.append(current_result)

        # Check for skip changes
        if (baseline_result.status == TestStatus.SKIPPED) != (
            current_result.status == TestStatus.SKIPPED
        ):
            skip_changes.append(current_result)

    return {
        "regressions": regressions,
        "fixes": fixes,
        "skip_changes": skip_changes,
        "added": [current_results[t] for t in added_tests],
        "removed": [baseline_results[t] for t in removed_tests],
    }


def format_json_output(
    baseline: PytestResults,
    current: PytestResults,
    differences: Dict[str, List[TestResult]],
) -> dict:
    """Format test results as JSON."""
    output = {
        "format_version": "1.0",
        "baseline": {
            "file": str(baseline.xml_path),
            "summary": baseline.get_summary(),
        },
        "current": {
            "file": str(current.xml_path),
            "summary": current.get_summary(),
        },
        "differences": {
            "regressions": [t.to_dict() for t in differences["regressions"]],
            "fixes": [t.to_dict() for t in differences["fixes"]],
            "skip_changes": [t.to_dict() for t in differences["skip_changes"]],
            "added": [t.to_dict() for t in differences["added"]],
            "removed": [t.to_dict() for t in differences["removed"]],
        },
    }
    return output


def format_console_output(
    baseline: PytestResults,
    current: PytestResults,
    differences: Dict[str, List[TestResult]],
    verbosity: int = 1,
):
    """Print differences in console format."""
    if not any(differences.values()):
        click.echo("No differences found between the test results.")
        return

    # Print summaries
    click.echo("\nBaseline Summary:")
    for key, value in baseline.get_summary().items():
        click.echo(f"  {key}: {value}")

    click.echo("\nCurrent Summary:")
    for key, value in current.get_summary().items():
        click.echo(f"  {key}: {value}")

    # Print regressions
    if differences["regressions"]:
        click.echo(
            "\nRegressions (tests that passed in baseline but failed in current):"
        )
        for test in differences["regressions"]:
            if verbosity >= 2:
                click.echo(f"\n{test}")
            else:
                click.echo(f"  {test.get_full_name()}")

    # Print fixes
    if differences["fixes"]:
        click.echo(
            "\nFixes (tests that failed in baseline but passed in current):"
        )
        for test in differences["fixes"]:
            if verbosity >= 2:
                click.echo(f"\n{test}")
            else:
                click.echo(f"  {test.get_full_name()}")

    # Print skip changes
    if differences["skip_changes"]:
        click.echo(
            "\nSkip changes (tests that were skipped in one file but not the other):"
        )
        for test in differences["skip_changes"]:
            if verbosity >= 2:
                click.echo(f"\n{test}")
            else:
                click.echo(f"  {test.get_full_name()}")

    # Print added tests
    if differences["added"]:
        click.echo("\nAdded tests (only in current):")
        for test in differences["added"]:
            if verbosity >= 2:
                click.echo(f"\n{test}")
            else:
                click.echo(f"  {test.get_full_name()}")

    # Print removed tests
    if differences["removed"]:
        click.echo("\nRemoved tests (only in baseline):")
        for test in differences["removed"]:
            if verbosity >= 2:
                click.echo(f"\n{test}")
            else:
                click.echo(f"  {test.get_full_name()}")

    # Print summary
    click.echo("\nSummary:")
    click.echo(f"  Regressions: {len(differences['regressions'])}")
    click.echo(f"  Fixes: {len(differences['fixes'])}")
    click.echo(f"  Skip changes: {len(differences['skip_changes'])}")
    click.echo(f"  Added tests: {len(differences['added'])}")
    click.echo(f"  Removed tests: {len(differences['removed'])}")


def format_markdown_output(
    baseline: PytestResults,
    current: PytestResults,
    differences: Dict[str, List[TestResult]],
    verbosity: int = 1,
):
    """Print differences in markdown format."""
    if not any(differences.values()):
        click.echo("No differences found between the test results.")
        return

    # Print combined summary table
    click.echo("\n## Test Summaries\n")
    click.echo("|Metric|Baseline|Current|")
    click.echo("|------|--------|--------|")

    baseline_summary = baseline.get_summary()
    current_summary = current.get_summary()

    for key in baseline_summary.keys():
        baseline_value = baseline_summary[key]
        current_value = current_summary[key]
        if isinstance(baseline_value, float):
            click.echo(f"|{key}|{baseline_value:.3f}|{current_value:.3f}|")
        else:
            click.echo(f"|{key}|{baseline_value}|{current_value}|")

    # Print regressions
    if differences["regressions"]:
        click.echo("\n## Regressions")
        click.echo("Tests that passed in baseline but failed in current:\n")
        for test in differences["regressions"]:
            if verbosity >= 2:
                click.echo(f"### {test.get_full_name()}")
                click.echo("```")
                click.echo(f"Error: {test.message}")
                if test.details:
                    click.echo("\nDetails:")
                    click.echo(test.details)
                click.echo("```\n")
            else:
                click.echo(f"- {test.get_full_name()}")

    # Print fixes
    if differences["fixes"]:
        click.echo("\n<details>")
        click.echo(
            "<summary> Tests that failed in baseline but passed in current </summary>\n"
        )
        click.echo("```")
        for test in differences["fixes"]:
            if verbosity >= 2:
                click.echo(f"{test.get_full_name()}")
                click.echo(f"Time: {test.time:.3f}s")
                if test.message:
                    click.echo(f"Previous error: {test.message}")
                if test.details:
                    click.echo(f"Previous details: {test.details}")
                click.echo("---")
            else:
                click.echo(test.get_full_name())
        click.echo("```")
        click.echo("</details>")

    # Print skip changes
    if differences["skip_changes"]:
        click.echo("\n## Skip Changes")
        click.echo("Tests that were skipped in one file but not the other:\n")
        for test in differences["skip_changes"]:
            if verbosity >= 2:
                click.echo(f"### {test.get_full_name()}")
                click.echo("```")
                click.echo(f"Error: {test.message}")
                if test.details:
                    click.echo("\nDetails:")
                    click.echo(test.details)
                click.echo("```\n")
            else:
                click.echo(f"- {test.get_full_name()}")

    # Print added tests
    if differences["added"]:
        click.echo("\n## Added Tests")
        click.echo("Tests that only exist in current:\n")
        for test in differences["added"]:
            if verbosity >= 2:
                click.echo(f"### {test.get_full_name()}")
                click.echo("```")
                click.echo(f"Error: {test.message}")
                if test.details:
                    click.echo("\nDetails:")
                    click.echo(test.details)
                click.echo("```\n")
            else:
                click.echo(f"- {test.get_full_name()}")

    # Print removed tests
    if differences["removed"]:
        click.echo("\n## Removed Tests")
        click.echo("Tests that only exist in baseline:\n")
        for test in differences["removed"]:
            if verbosity >= 2:
                click.echo(f"### {test.get_full_name()}")
                click.echo("```")
                click.echo(f"Error: {test.message}")
                if test.details:
                    click.echo("\nDetails:")
                    click.echo(test.details)
                click.echo("```\n")
            else:
                click.echo(f"- {test.get_full_name()}")

    # Print summary
    click.echo("\n## Summary")
    click.echo("|Category|Count|")
    click.echo("|--------|-----|")
    click.echo(f"|Regressions|{len(differences['regressions'])}|")
    click.echo(f"|Fixes|{len(differences['fixes'])}|")
    click.echo(f"|Skip Changes|{len(differences['skip_changes'])}|")
    click.echo(f"|Added Tests|{len(differences['added'])}|")
    click.echo(f"|Removed Tests|{len(differences['removed'])}|")


def format_pytest_output(
    differences: Dict[str, List[TestResult]], select: str = "fixes"
) -> str:
    """Format test results as a list of test paths for pytest selection.

    Returns a string containing test paths separated by newlines that can be passed
    directly to pytest. Each test path is prefixed with "sklearn." to match the test
    organization.

    Args:
        differences: Dictionary containing test result differences
        select: Which tests to include ("regressions", "fixes", "skip_changes", "all")
    """

    def format_test_path(classname: str, name: str) -> str:
        """Format a single test path."""
        # Remove any existing sklearn prefix to avoid duplication
        if classname.startswith("sklearn."):
            classname = classname[8:]  # Remove "sklearn." prefix
        return f"sklearn.{classname}::{name}"

    # Collect all test paths that changed
    test_paths = []

    # Add tests based on selection
    if select in ["regressions", "all"]:
        for test in differences["regressions"]:
            test_paths.append(format_test_path(test.classname, test.name))

    if select in ["fixes", "all"]:
        for test in differences["fixes"]:
            test_paths.append(format_test_path(test.classname, test.name))

    if select in ["skip_changes", "all"]:
        for test in differences["skip_changes"]:
            test_paths.append(format_test_path(test.classname, test.name))

    if select == "all":
        for test in differences["added"] + differences["removed"]:
            test_paths.append(format_test_path(test.classname, test.name))

    # Return test paths separated by newlines
    return "\n".join(test_paths)


@click.command()
@click.argument("baseline_xml", type=click.Path(exists=True, path_type=Path))
@click.argument("current_xml", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["brief", "normal", "detailed"]),
    default="normal",
    help="Output verbosity level",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Write output to file instead of stdout",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json", "markdown", "pytest"]),
    default="console",
    help="Output format (console, json, markdown, or pytest)",
)
@click.option(
    "--select",
    type=click.Choice(["regressions", "fixes", "skip_changes", "all"]),
    default="fixes",
    help="Which tests to include when using --format=pytest",
)
def main(
    baseline_xml: Path,
    current_xml: Path,
    verbosity: str,
    output: Optional[Path],
    output_format: str,
    select: str,
):
    """Compare two pytest XML results files and show differences in test outcomes.

    BASELINE_XML: Path to the baseline pytest XML results file
    CURRENT_XML: Path to the current pytest XML results file
    """
    try:
        # Parse both XML files
        baseline = PytestResults(baseline_xml)
        current = PytestResults(current_xml)

        # Redirect output if specified
        if output:
            sys.stdout = open(output, "w")

        # Analyze differences
        differences = analyze_differences(baseline, current)
        verb_level = {"brief": 0, "normal": 1, "detailed": 2}[verbosity]

        # Output in requested format
        if output_format == "json":
            output_data = format_json_output(baseline, current, differences)
            click.echo(json.dumps(output_data, indent=2))
        elif output_format == "pytest":
            pytest_expr = format_pytest_output(differences, select)
            click.echo(pytest_expr)
        elif output_format == "markdown":
            format_markdown_output(baseline, current, differences, verb_level)
        else:  # console
            format_console_output(baseline, current, differences, verb_level)

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
