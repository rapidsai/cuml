#!/usr/bin/env python3
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

"""Summarize test results from a JUnit XML report file."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


class QuoteTestID(str):
    """String subclass to force quoting of test IDs."""

    pass


def setup_yaml():
    """Configure YAML dumper with custom string handling."""

    def quoted_scalar(dumper, data):
        scalar_tag = "tag:yaml.org,2002:str"
        return dumper.represent_scalar(scalar_tag, data, style='"')

    yaml.add_representer(QuoteTestID, quoted_scalar)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize test results from a JUnit XML report file"
    )
    parser.add_argument(
        "report_file",
        type=Path,
        help="Path to the JUnit XML report file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed failure information",
    )
    parser.add_argument(
        "-f",
        "--fail-below",
        type=float,
        default=0.0,
        help="Minimum pass rate threshold [0-100] (default: 0)",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "xfail_list"],
        default="summary",
        help="Output format (default: summary)",
    )
    return parser.parse_args()


def validate_threshold(threshold):
    """Validate that the threshold is between 0 and 100."""
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")


def main():
    """Main entry point."""
    args = parse_args()
    validate_threshold(args.fail_below)

    if not args.report_file.exists():
        print(f"Error: Report file not found: {args.report_file}")
        sys.exit(1)

    try:
        tree = ET.parse(args.report_file)
    except ET.ParseError as e:
        print(f"Error: Invalid XML file: {e}")
        sys.exit(1)

    root = tree.getroot()
    testsuite = root.find("testsuite")
    if testsuite is None:
        print("Error: No testsuite element found in XML file")
        sys.exit(1)

    # Extract test statistics
    total_tests = int(testsuite.get("tests", 0))
    failures = int(testsuite.get("failures", 0))
    errors = int(testsuite.get("errors", 0))
    skipped = int(testsuite.get("skipped", 0))
    time = float(testsuite.get("time", 0))

    # Count xfailed tests
    xfailed = 0
    for testcase in testsuite.findall(".//testcase"):
        failure = testcase.find("failure")
        if failure is not None and failure.get("message") == "xfail":
            xfailed += 1
    regular_failures = failures - xfailed

    # Calculate passed tests and pass rate
    passed = total_tests - failures - errors - skipped
    pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0

    if args.format == "xfail_list":
        # Generate xfail list entries for failed tests
        xfail_list = []
        for testcase in testsuite.findall(".//testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                if failure is not None and failure.get("message") == "xfail":
                    continue  # Skip already xfailed tests
                # Ensure test ID includes sklearn prefix
                classname = testcase.get("classname", "")
                if not classname.startswith("sklearn."):
                    classname = f"sklearn.{classname}"
                test_id = f"{classname}::{testcase.get('name')}"
                # Only quote the test ID value, not the key
                xfail_list.append({"id": QuoteTestID(test_id)})
        # Sort entries alphabetically by test ID
        xfail_list.sort(key=lambda x: x["id"])
        # Use a large width to prevent unwanted line breaks
        setup_yaml()
        print(yaml.dump(xfail_list, sort_keys=False, width=float("inf")))
        return

    # Print summary
    print("Test Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {regular_failures}")
    print(f"  XFailed:     {xfailed}")
    print(f"  Errors:      {errors}")
    print(f"  Skipped:     {skipped}")
    print(f"  Pass Rate:   {pass_rate:.2f}%")
    print(f"  Total Time:  {time:.2f}s")

    # List failed tests in verbose mode
    if (failures + errors) > 0 and args.verbose:
        print("\nFailed Tests:")
        for testcase in testsuite.findall(".//testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                if failure is not None and failure.get("message") == "xfail":
                    print(f"  {testcase.get('name')} (xfail)")
                else:
                    print(f"  {testcase.get('name')}")

    # Check threshold
    if pass_rate < args.fail_below:
        print(
            f"\nError: Pass rate {pass_rate:.2f}% is below threshold "
            f"{args.fail_below}%"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
