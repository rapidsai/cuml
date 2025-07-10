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
from collections import OrderedDict
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
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping(
            "tag:yaml.org,2002:map", data.items()
        ),
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


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
        "-c",
        "--config",
        type=Path,
        help="Path to config file",
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
        help="Minimum pass rate threshold [0-100] (default: 0)",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "xfail_list", "traceback"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--update-xfail-list",
        type=Path,
        help="Path to existing xfail list to update",
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Update the xfail list file in place",
    )
    parser.add_argument(
        "--xpassed",
        choices=["keep", "remove", "mark-flaky"],
        default="keep",
        help="How to handle XPASS tests (default: keep)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit output to first N entries (default: no limit)",
    )
    args = parser.parse_args()

    # Handle fail-below threshold logic:
    # 1. Use command line value if provided
    # 2. Use config value if no command line value
    # 3. Use default of 0.0 if neither is provided
    if args.fail_below is None:
        if args.config is not None:
            config = load_config(args.config)
            args.fail_below = config.get("threshold", {}).get(
                "fail_below", 0.0
            )
        else:
            args.fail_below = 0.0

    return args


def validate_threshold(threshold):
    """Validate that the threshold is between 0 and 100."""
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")


def load_existing_xfail_list(path):
    """Load existing xfail list from file."""
    if not path.exists():
        return []
    with open(path) as f:
        return yaml.safe_load(f) or []


def update_xfail_list(existing_list, test_results, xpassed_action="keep"):
    """Update existing xfail list based on test results.

    Args:
        existing_list: List of existing xfail groups
        test_results: Dict containing test results by test ID
        xpassed_action: How to handle XPASS tests
            ("keep", "remove", or "mark-flaky")

    Returns:
        Updated xfail list
    """
    # Convert existing list to dict for easier lookup
    existing_by_id = {}
    for group in existing_list:
        for test_id in group.get("tests", []):
            existing_by_id[QuoteTestID(test_id)] = group

    updated_groups = {}

    # First process existing groups
    for group in existing_list:
        reason = group["reason"]
        strict = group.get("strict", True)
        tests = group.get("tests", [])
        condition = group.get("condition", None)
        marker = group.get("marker", None)

        # Track which tests from this group are still relevant
        relevant_tests = []

        for test_id in tests:
            test_id = QuoteTestID(test_id)
            if test_id in test_results:
                result = test_results[test_id]
                if result["status"] in ("fail", "xfail"):
                    # Keep failing tests
                    relevant_tests.append(test_id)
                elif result["status"] == "xpass":
                    if xpassed_action == "keep":
                        # Keep all xpassed tests
                        relevant_tests.append(test_id)
                    elif xpassed_action == "mark-flaky" and strict:
                        # Move strict xpassed tests to flaky group instead of
                        # modifying their existing group
                        flaky_reason = "Test is flaky with cuml.accel"
                        if flaky_reason not in updated_groups:
                            updated_groups[flaky_reason] = OrderedDict(
                                [
                                    ("reason", flaky_reason),
                                    ("strict", False),
                                    ("tests", []),
                                ]
                            )
                            if condition is not None:
                                updated_groups[flaky_reason][
                                    "condition"
                                ] = condition
                            if marker is not None:
                                updated_groups[flaky_reason]["marker"] = marker
                        updated_groups[flaky_reason]["tests"].append(test_id)
                    # For "remove", we don't add xpassed tests at all
                elif not strict:
                    # Always keep non-strict tests
                    relevant_tests.append(test_id)
            else:
                # Test not in results - keep it to be safe
                relevant_tests.append(test_id)

        if relevant_tests:
            if reason not in updated_groups:
                updated_groups[reason] = OrderedDict(
                    [
                        ("reason", reason),
                        ("strict", strict),
                        ("tests", []),
                    ]
                )
                if condition is not None:
                    updated_groups[reason]["condition"] = condition
                if marker is not None:
                    updated_groups[reason]["marker"] = marker
            updated_groups[reason]["tests"].extend(relevant_tests)

    # Then add any new failing tests
    for test_id, result in test_results.items():
        if test_id not in existing_by_id and result["status"] in (
            "fail",
            "xfail",
        ):
            # Create a new group for this test
            reason = "Test fails with cuml.accel"
            if reason not in updated_groups:
                updated_groups[reason] = OrderedDict(
                    [
                        ("reason", reason),
                        ("strict", True),
                        ("tests", []),
                    ]
                )
            updated_groups[reason]["tests"].append(test_id)

    # Convert back to list and sort
    final_groups = []
    for group in sorted(updated_groups.values(), key=lambda x: x["reason"]):
        group["tests"] = sorted(group["tests"])
        final_groups.append(group)
    return final_groups


def get_test_results(testsuite):
    """Extract test results from testsuite.

    Returns dict mapping test IDs to their results.
    """
    results = {}
    for testcase in testsuite.findall(".//testcase"):
        test_id = QuoteTestID(get_test_id(testcase))

        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped_elem = testcase.find("skipped")

        if failure is not None:
            msg = str(failure.get("message"))
            if "XPASS(strict)" in msg:
                status = "xpass"
            elif msg == "xfail":
                status = "xfail"
            else:
                status = "fail"
        elif error is not None:
            if "XPASS(strict)" in str(error.get("message")):
                status = "xpass"
            else:
                status = "fail"
        elif (
            skipped_elem is not None
            and skipped_elem.get("type") == "pytest.xfail"
        ):
            status = "xfail"
        else:
            status = "pass"

        results[test_id] = {
            "status": status,
        }

    return results


def format_table(rows, col_sep="  "):
    """Format a table with aligned columns.

    Args:
        rows: List of rows, where each row is a list of strings
        col_sep: String to separate columns

    Returns:
        List of formatted row strings
    """
    if not rows:
        return []

    # Calculate column widths
    num_cols = len(rows[0])
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format each row
    formatted_rows = []
    for row in rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            # Right-align numeric values, left-align text
            if i > 0 and cell.replace(".", "").replace("%", "").isdigit():
                formatted_cells.append(f"{cell:>{col_widths[i]}}")
            else:
                formatted_cells.append(f"{cell:<{col_widths[i]}}")
        formatted_rows.append(col_sep.join(formatted_cells))

    return formatted_rows


def get_test_id(testcase) -> str:
    classname = testcase.get("classname", "")
    name = testcase.get("name")
    return f"{classname}::{name}" if classname else name


def format_traceback_output(testsuite, limit=None):
    """Format test results showing tracebacks of failed tests.

    Args:
        testsuite: XML testsuite element containing test results
        limit: Optional limit on number of entries to show

    Returns:
        List of formatted strings containing test results and tracebacks
    """
    output = []
    output.append("Failed Tests with Tracebacks:")
    output.append("=" * 80)

    count = 0
    for testcase in testsuite.findall(".//testcase"):
        if limit is not None and count >= limit:
            output.append(f"\n... (showing first {limit} entries)")
            break

        failure = testcase.find("failure")
        error = testcase.find("error")

        if failure is not None or error is not None:
            test_id = get_test_id(testcase)

            msg = ""
            details = ""

            if failure is not None:
                msg = failure.get("message", "")
                details = failure.text
            elif error is not None:
                msg = error.get("message", "")
                details = error.text

            if "XPASS" in msg:
                continue  # Skip xpassed tests
            elif msg == "xfail":
                continue  # Skip xfailed tests

            output.append(f"\nTest: {test_id}")
            output.append("-" * 80)
            if msg:
                output.append(f"Error: {msg}")
            if details:
                output.append("\nTraceback:")
                output.append(details)
            output.append("=" * 80)
            count += 1

    return output


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
    total_errors = int(testsuite.get("errors", 0))
    total_skipped = int(testsuite.get("skipped", 0))
    time = float(testsuite.get("time", 0))

    # Count failures, xfails, and xpasses separately
    regular_failures = 0
    xfailed = 0
    xpassed_strict = 0
    xpassed_non_strict = 0
    for testcase in testsuite.findall(".//testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped_elem = testcase.find("skipped")

        if failure is not None:
            msg = str(failure.get("message"))
            if "XPASS(strict)" in msg:
                xpassed_strict += 1
            elif "XPASS" in msg:
                xpassed_non_strict += 1
            elif msg == "xfail":
                xfailed += 1
            else:
                regular_failures += 1
        elif error is not None:
            msg = str(error.get("message"))
            if "XPASS(strict)" in msg:
                xpassed_strict += 1
            elif "XPASS" in msg:
                xpassed_non_strict += 1
            else:
                regular_failures += 1
        elif (
            skipped_elem is not None
            and skipped_elem.get("type") == "pytest.xfail"
        ):
            xfailed += 1

    # Calculate passed tests and pass rate
    passed = (
        total_tests
        - regular_failures
        - xfailed
        - xpassed_strict
        - xpassed_non_strict
        - total_errors
        - total_skipped
    )
    pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0

    if args.format == "traceback":
        output = format_traceback_output(testsuite, args.limit)
        print("\n".join(output))
        return

    if args.format == "xfail_list" or args.update_xfail_list:
        # Get test results
        test_results = get_test_results(testsuite)

        if args.update_xfail_list:
            if not args.update_xfail_list.exists():
                print(f"Error: Xfail list not found: {args.update_xfail_list}")
                sys.exit(1)
            # Update existing xfail list
            existing_list = load_existing_xfail_list(args.update_xfail_list)
            xfail_list = update_xfail_list(
                existing_list, test_results, args.xpassed
            )
            # Write to file if in-place, otherwise print
            setup_yaml()
            if args.in_place:
                with open(args.update_xfail_list, "w") as f:
                    yaml.dump(
                        xfail_list, f, sort_keys=False, width=float("inf")
                    )
                print(f"Updated {args.update_xfail_list}")
            else:
                print(
                    yaml.dump(xfail_list, sort_keys=False, width=float("inf"))
                )
        else:
            # Generate new xfail list
            xfail_list = []
            count = 0
            for test_id, result in test_results.items():
                if args.limit is not None and count >= args.limit:
                    break
                if result["status"] in ("fail", "xfail"):
                    if not xfail_list:
                        xfail_list.append(
                            OrderedDict(
                                [
                                    ("reason", "Test fails with cuml.accel"),
                                    ("strict", True),
                                    ("tests", []),
                                ]
                            )
                        )
                    xfail_list[0]["tests"].append(test_id)
                    count += 1
            if xfail_list:
                xfail_list[0]["tests"].sort()
            # Print to stdout
            setup_yaml()
            print(yaml.dump(xfail_list, sort_keys=False, width=float("inf")))
        return

    # Print summary
    print("Test Summary:")
    rows = [
        ["Total Tests:", str(total_tests)],
        ["Passed:", str(passed)],
        ["Failed:", str(regular_failures)],
        ["XFailed:", str(xfailed)],
        ["XPassed (strict):", str(xpassed_strict)],
        ["XPassed (non-strict):", str(xpassed_non_strict)],
        ["Errors:", str(total_errors)],
        ["Skipped:", str(total_skipped)],
        ["Pass Rate:", f"{pass_rate:.2f}%"],
        ["Total Time:", f"{time:.2f}s"],
    ]
    for row in format_table(rows, "  "):
        print(f"  {row}")

    # List failed tests in verbose mode
    if (regular_failures + total_errors) > 0 and args.verbose:
        print("\nFailed Tests:")
        count = 0
        for testcase in testsuite.findall(".//testcase"):
            if args.limit is not None and count >= args.limit:
                print(f"  ... (showing first {args.limit} entries)")
                break
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                test_id = get_test_id(testcase)
                msg = ""
                if failure is not None and failure.get("message") is not None:
                    msg = failure.get("message")
                elif error is not None and error.get("message") is not None:
                    msg = error.get("message")
                if "XPASS" in msg:
                    continue  # Skip xpassed tests in failure list
                elif msg == "xfail":
                    print(f"  {test_id} (xfail)")
                else:
                    print(f"  {test_id}")
                count += 1

    # List strict xpasses in verbose mode
    if xpassed_strict > 0 and args.verbose:
        print("\nPotential Improvements (Strict XPASS):")
        count = 0
        for testcase in testsuite.findall(".//testcase"):
            if args.limit is not None and count >= args.limit:
                print(f"  ... (showing first {args.limit} entries)")
                break
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                msg = ""
                if failure is not None and failure.get("message") is not None:
                    msg = failure.get("message")
                elif error is not None and error.get("message") is not None:
                    msg = error.get("message")
                if "XPASS(strict)" in msg:
                    test_id = get_test_id(testcase)
                    print(f"  {test_id}")
                    count += 1

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
