#!/usr/bin/env python3
#
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
#
"""
Xfail list management tool for cuml.accel tests.

This module provides tools for editing and maintaining xfail lists used in
cuml.accel testing. It offers both a programmatic API through the XfailManager
and XfailGroup classes, and a command-line interface for common operations.

Key features:
- Edit test properties and move tests between groups
- Add/remove tests from existing groups
- Deterministic formatting and sorting of xfail lists
- Support for @file syntax to read test IDs from files
- Validation of test IDs and group conditions

CLI Commands:
- edit-tests: Modify properties of tests and move them to new groups
- add-tests: Add tests to existing groups
- remove-tests: Remove tests from xfail lists
- format: Apply consistent formatting and sorting

The tool ensures xfail lists remain maintainable and produce clean diffs
in version control systems.
"""

import argparse
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from packaging.requirements import Requirement


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


class XfailGroup:
    """Represents a group of related test failures with common characteristics."""

    def __init__(
        self,
        reason: str,
        tests: Optional[List[str]] = None,
        strict: bool = True,
        condition: Optional[str] = None,
        run: bool = True,
        marker: Optional[str] = None,
    ):
        """Initialize an xfail group.

        Args:
            reason: Description of why tests in this group are expected to fail
            tests: List of test IDs in format "module::test_name"
            strict: Whether to enforce xfail (default: True)
            condition: Optional version requirement (e.g., "scikit-learn>=1.5.2")
            marker: Optional pytest marker for grouping tests
        """
        self.reason = reason
        self.tests = [QuoteTestID(test) for test in (tests or [])]
        self.strict = strict
        self.condition = condition
        self.run = run
        self.marker = marker

    def add_test(self, test_id: str) -> None:
        """Add a test ID to this group."""
        quoted_id = QuoteTestID(test_id)
        if quoted_id not in self.tests:
            self.tests.append(quoted_id)

    def remove_test(self, test_id: str) -> bool:
        """Remove a test ID from this group.

        Returns:
            True if test was removed, False if it wasn't in the group
        """
        try:
            self.tests.remove(QuoteTestID(test_id))
            return True
        except ValueError:
            return False

    def has_test(self, test_id: str) -> bool:
        """Check if a test ID is in this group."""
        return QuoteTestID(test_id) in self.tests

    def to_dict(self) -> OrderedDict:
        """Convert to OrderedDict format for YAML serialization."""
        result = OrderedDict(
            [
                ("reason", self.reason),
            ]
        )

        # Add optional fields in order: marker, condition, strict, run, tests
        if self.marker:
            result["marker"] = self.marker
        if self.condition:
            result["condition"] = self.condition

        # Only include strict if it's False (to match summarize-results.py)
        if not self.strict:
            result["strict"] = self.strict

        # Only include run if it's False (default is True)
        if not self.run:
            result["run"] = self.run

        result["tests"] = sorted(self.tests)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XfailGroup":
        """Create XfailGroup from dictionary."""
        return cls(
            reason=data["reason"],
            tests=[QuoteTestID(test) for test in data.get("tests", [])],
            strict=data.get("strict", True),
            condition=data.get("condition"),
            run=data.get("run", True),
            marker=data.get("marker"),
        )

    def __len__(self) -> int:
        """Return number of tests in this group."""
        return len(self.tests)

    def __bool__(self) -> bool:
        """Return True if group has tests."""
        return bool(self.tests)

    def __lt__(self, other: "XfailGroup") -> bool:
        """Define ordering for deterministic sorting of groups."""
        if not isinstance(other, XfailGroup):
            return NotImplemented

        # Primary sort: by reason (alphabetically)
        if self.reason != other.reason:
            return self.reason < other.reason

        # Secondary sort: by marker (None values last)
        self_marker = self.marker or "zzz_no_marker"
        other_marker = other.marker or "zzz_no_marker"
        if self_marker != other_marker:
            return self_marker < other_marker

        # Tertiary sort: by condition (None values last)
        self_condition = self.condition or "zzz_no_condition"
        other_condition = other.condition or "zzz_no_condition"
        if self_condition != other_condition:
            return self_condition < other_condition

        # Quaternary sort: by strict (strict groups first)
        if self.strict != other.strict:
            return self.strict > other.strict  # True > False

        # Quinary sort: by run (run groups first)
        if self.run != other.run:
            return self.run > other.run  # True > False

        # Final sort: by number of tests (fewer tests first)
        return len(self.tests) < len(other.tests)


class XfailManager:
    """Manager for xfail lists with programmatic API."""

    def __init__(self, xfail_list_path: Optional[Union[str, Path]] = None):
        """Initialize XfailManager.

        Args:
            xfail_list_path: Path to existing xfail list file, or None for
                empty list
        """
        self.groups: List[XfailGroup] = []
        self._test_to_groups: Dict[str, List[XfailGroup]] = defaultdict(list)

        if xfail_list_path:
            self.load(xfail_list_path)

    def load(self, xfail_list_path: Union[str, Path]) -> None:
        """Load xfail list from YAML file."""
        path = Path(xfail_list_path)
        if not path.exists():
            raise FileNotFoundError(f"Xfail list file not found: {path}")

        data = yaml.safe_load(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Xfail list must be a list of test groups")

        self.groups = [XfailGroup.from_dict(group_data) for group_data in data]
        self._rebuild_test_index()

    def save(self, xfail_list_path: Union[str, Path]) -> None:
        """Save xfail list to YAML file."""
        path = Path(xfail_list_path)

        # Filter out empty groups and sort deterministically
        non_empty_groups = [group for group in self.groups if group]
        sorted_groups = sorted(non_empty_groups)

        data = [group.to_dict() for group in sorted_groups]

        # Setup YAML formatting to match summarize-results.py
        setup_yaml()
        with path.open("w") as f:
            yaml.dump(data, f, sort_keys=False, width=float("inf"))

    def add_group(self, group: XfailGroup) -> None:
        """Add an xfail group."""
        self.groups.append(group)
        for test_id in group.tests:
            self._test_to_groups[test_id].append(group)

    def create_group(
        self,
        reason: str,
        tests: Optional[List[str]] = None,
        strict: bool = True,
        run: bool = True,
        condition: Optional[str] = None,
        marker: Optional[str] = None,
    ) -> XfailGroup:
        """Create and add a new xfail group.

        Returns:
            The created XfailGroup
        """
        group = XfailGroup(reason, tests, strict, condition, marker)
        self.add_group(group)
        return group

    def remove_group(self, group: XfailGroup) -> bool:
        """Remove an xfail group.

        Returns:
            True if group was removed, False if it wasn't found
        """
        try:
            self.groups.remove(group)
            self._rebuild_test_index()
            return True
        except ValueError:
            return False

    def find_groups_by_marker(self, marker: str) -> List[XfailGroup]:
        """Find all groups with a specific marker."""
        return [group for group in self.groups if group.marker == marker]

    def find_groups_by_reason_pattern(self, pattern: str) -> List[XfailGroup]:
        """Find groups whose reason matches a regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [group for group in self.groups if regex.search(group.reason)]

    def get_test_groups(self, test_id: str) -> List[XfailGroup]:
        """Get all groups that contain a specific test ID."""
        quoted_id = QuoteTestID(test_id)
        return self._test_to_groups.get(quoted_id, [])

    def add_test_to_group(self, test_id: str, group: XfailGroup) -> None:
        """Add a test to an existing group."""
        quoted_id = QuoteTestID(test_id)
        if not group.has_test(test_id):
            group.add_test(test_id)
            self._test_to_groups[quoted_id].append(group)

    def remove_test_from_group(self, test_id: str, group: XfailGroup) -> bool:
        """Remove a test from a specific group.

        Returns:
            True if test was removed, False if it wasn't in the group
        """
        quoted_id = QuoteTestID(test_id)
        if group.remove_test(test_id):
            self._test_to_groups[quoted_id].remove(group)
            return True
        return False

    def remove_test_completely(self, test_id: str) -> int:
        """Remove a test from all groups.

        Returns:
            Number of groups the test was removed from
        """
        quoted_id = QuoteTestID(test_id)
        groups = self._test_to_groups.get(quoted_id, []).copy()
        count = 0
        for group in groups:
            if self.remove_test_from_group(test_id, group):
                count += 1
        return count

    def move_test(
        self, test_id: str, from_group: XfailGroup, to_group: XfailGroup
    ) -> bool:
        """Move a test from one group to another.

        Returns:
            True if test was moved, False if it wasn't in from_group
        """
        if self.remove_test_from_group(test_id, from_group):
            self.add_test_to_group(test_id, to_group)
            return True
        return False

    def cleanup_empty_groups(self) -> int:
        """Remove groups with no tests.

        Returns:
            Number of groups removed
        """
        empty_groups = [group for group in self.groups if not group]
        for group in empty_groups:
            self.remove_group(group)
        return len(empty_groups)

    def validate_conditions(self) -> List[str]:
        """Validate all version conditions in groups.

        Returns:
            List of error messages for invalid conditions
        """
        errors = []
        for i, group in enumerate(self.groups):
            if group.condition:
                try:
                    Requirement(group.condition)
                except Exception as e:
                    errors.append(
                        f"Group {i} has invalid condition "
                        f"'{group.condition}': {e}"
                    )
        return errors

    def _rebuild_test_index(self) -> None:
        """Rebuild the internal test-to-groups index."""
        self._test_to_groups.clear()
        for group in self.groups:
            for test_id in group.tests:
                quoted_id = QuoteTestID(test_id)
                self._test_to_groups[quoted_id].append(group)


# CLI Commands Implementation


def parse_test_ids(test_args: List[str]) -> List[str]:
    """Parse test IDs, expanding @file references.

    Args:
        test_args: List of test ID arguments, may include @filename entries

    Returns:
        List of test IDs with @file references expanded
    """
    test_ids = []

    for arg in test_args:
        if arg.startswith("@"):
            # Read test IDs from file
            filename = arg[1:]  # Remove @ prefix
            try:
                with open(filename) as f:
                    file_tests = [line.strip() for line in f if line.strip()]
                    test_ids.extend(file_tests)
            except FileNotFoundError:
                print(f"Error: File not found: {filename}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading file {filename}: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Regular test ID
            test_ids.append(arg)

    return test_ids


def cmd_edit(args):
    """Edit properties of tests and move them to new groups."""
    manager = XfailManager(args.xfail_list)

    test_ids = parse_test_ids(args.tests)

    # Check for conflicting flags
    if args.strict and args.non_strict:
        print(
            "Error: Cannot specify both --strict and --non-strict",
            file=sys.stderr,
        )
        return 1

    if args.run and args.no_run:
        print(
            "Error: Cannot specify both --run and --no-run",
            file=sys.stderr,
        )
        return 1

    # Validate all test IDs exist
    missing_tests = []
    for test_id in test_ids:
        existing_groups = manager.get_test_groups(test_id)
        if not existing_groups:
            missing_tests.append(test_id)

    if missing_tests:
        print(
            f"Tests not found in xfail list: {missing_tests}", file=sys.stderr
        )
        return 1

    # Check if any properties actually need to change
    if not (
        args.reason
        or args.strict
        or args.non_strict
        or args.run
        or args.no_run
        or args.condition is not None
        or args.marker is not None
    ):
        print("No changes specified - tests remain in same groups")
        return 0

    # Process each test ID
    moved_count = 0
    empty_groups_to_remove = set()

    for test_id in test_ids:
        existing_groups = manager.get_test_groups(test_id)
        if not existing_groups:
            continue  # Already validated above, but be safe

        source_group = existing_groups[0]

        # Determine new properties (use provided values or inherit from source)
        new_reason = args.reason if args.reason else source_group.reason

        # Handle strict flags
        if args.strict:
            new_strict = True
        elif args.non_strict:
            new_strict = False
        else:
            new_strict = source_group.strict  # Inherit from source

        new_condition = (
            args.condition
            if args.condition is not None
            else source_group.condition
        )
        new_marker = (
            args.marker if args.marker is not None else source_group.marker
        )

        # Handle run flags
        if args.run:
            new_run = True
        elif args.no_run:
            new_run = False
        else:
            new_run = source_group.run  # Inherit from source

        # Check if properties actually changed for this test
        properties_changed = (
            new_reason != source_group.reason
            or new_strict != source_group.strict
            or new_condition != source_group.condition
            or new_marker != source_group.marker
            or new_run != source_group.run
        )

        if not properties_changed:
            continue  # Skip this test, no changes needed

        # Check if a group with these exact properties already exists
        target_group = None
        for group in manager.groups:
            if (
                group.reason == new_reason
                and group.strict == new_strict
                and group.condition == new_condition
                and group.marker == new_marker
                and group.run == new_run
            ):
                target_group = group
                break

        # Create new group if needed
        if target_group is None:
            target_group = manager.create_group(
                reason=new_reason,
                run=new_run,
                tests=[],
                strict=new_strict,
                condition=new_condition,
                marker=new_marker,
            )
            print(f"Created new group: {new_reason}")

        # Move test from source to target group
        if manager.move_test(test_id, source_group, target_group):
            print(f"Moved test '{test_id}' to group with updated properties")
            moved_count += 1

            # Mark empty groups for removal
            if not source_group:
                empty_groups_to_remove.add(source_group)
        else:
            print(f"Failed to move test '{test_id}'", file=sys.stderr)

    # Clean up empty groups
    removed_groups = 0
    for empty_group in empty_groups_to_remove:
        if manager.remove_group(empty_group):
            removed_groups += 1

    if removed_groups > 0:
        print(f"Removed {removed_groups} empty groups")

    # Save the updated xfail list
    if args.in_place:
        manager.save(args.xfail_list)
    else:
        manager.save(args.output or args.xfail_list)

    print(f"Successfully processed {moved_count} tests")
    return 0


def cmd_add_tests(args):
    """Add tests to an existing group."""
    manager = XfailManager(args.xfail_list)

    # Find target group
    if args.marker:
        groups = manager.find_groups_by_marker(args.marker)
        if not groups:
            print(
                f"No group found with marker: {args.marker}", file=sys.stderr
            )
            return 1
        target_group = groups[0]
    elif args.reason_pattern:
        groups = manager.find_groups_by_reason_pattern(args.reason_pattern)
        if not groups:
            print(
                f"No group found matching pattern: {args.reason_pattern}",
                file=sys.stderr,
            )
            return 1
        target_group = groups[0]
    else:
        print(
            "Must specify either --marker or --reason-pattern", file=sys.stderr
        )
        return 1

    # Get tests to add
    tests = []
    if args.tests:
        tests = parse_test_ids(args.tests)

    # Add tests
    for test in tests:
        manager.add_test_to_group(test, target_group)

    if args.in_place:
        manager.save(args.xfail_list)
    else:
        manager.save(args.output or args.xfail_list)

    print(f"Added {len(tests)} tests to group: {target_group.reason}")
    return 0


def cmd_remove_tests(args):
    """Remove tests from xfail list."""
    manager = XfailManager(args.xfail_list)

    tests = []
    if args.tests:
        tests = parse_test_ids(args.tests)

    removed_count = 0
    for test in tests:
        count = manager.remove_test_completely(test)
        removed_count += count

    if args.in_place:
        manager.save(args.xfail_list)
    else:
        manager.save(args.output or args.xfail_list)

    print(f"Removed {removed_count} test entries")
    return 0


def cmd_format(args):
    """Auto-format xfail list(s) according to standards."""
    from pathlib import Path

    # Handle multiple files
    xfail_paths = [Path(p) for p in args.xfail_list]

    # Validate all files exist first
    for xfail_path in xfail_paths:
        if not xfail_path.exists():
            print(
                f"Error: Xfail list file not found: {xfail_path}",
                file=sys.stderr,
            )
            return 1

    # Track overall results
    total_errors = 0
    total_changes = 0

    for xfail_path in xfail_paths:
        if args.output and len(xfail_paths) > 1:
            print(
                "Error: --output cannot be used with multiple files",
                file=sys.stderr,
            )
            return 1

        result = _format_single_file(xfail_path, args)
        if result > 0:
            total_errors += 1
        elif result == 0 and not args.check:
            # In format mode, 0 means changes were made or no changes needed
            pass
        elif result == 0 and args.check:
            # In check mode, 0 means no changes needed
            pass
        else:
            # In check mode, 1 means changes needed
            total_changes += 1

    if args.check:
        # In check mode, return 1 if any files need changes
        return 1 if total_changes > 0 else 0
    else:
        # In format mode, return 1 if any files had errors
        return 1 if total_errors > 0 else 0


def _format_single_file(xfail_path, args):
    """Format a single xfail file.

    Returns 0 on success, 1 on error/changes needed.
    """
    import tempfile
    from pathlib import Path

    # Read original content
    original_content = xfail_path.read_text()

    try:
        # Load manager from file
        manager = XfailManager(xfail_path)

        # Apply cleanup if requested
        removed_groups = 0
        if args.cleanup:
            removed_groups = manager.cleanup_empty_groups()

        # Validate the list
        validation_errors = manager.validate_conditions()
        if validation_errors and not args.ignore_validation:
            if not args.quiet:
                print(f"Validation errors in {xfail_path}:", file=sys.stderr)
                for error in validation_errors:
                    print(f"  {error}", file=sys.stderr)
            if not args.fix_validation:
                return 1

        # Generate formatted content
        if args.check:
            # In check mode, save to temporary file to get formatted content
            # without modifying the original
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                manager.save(tmp_path)
                formatted_content = tmp_path.read_text()
                tmp_path.unlink()  # Clean up temp file
        else:
            # In format mode, save to the intended output location
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = xfail_path

            manager.save(output_path)
            formatted_content = output_path.read_text()

        # Check if changes were made
        content_changed = original_content != formatted_content

        if args.check:
            # Check mode: return 1 if formatting would change content
            if content_changed:
                print(f"File {xfail_path} needs formatting")
                return 1
            else:
                if not args.quiet:
                    print(f"File {xfail_path} is already properly formatted")
                return 0
        else:
            # Format mode: apply changes and report
            changes = []
            if content_changed:
                changes.append("formatting applied")
            if removed_groups > 0:
                changes.append(f"{removed_groups} empty groups removed")

            if changes:
                if not args.quiet:
                    output_path = (
                        Path(args.output) if args.output else xfail_path
                    )
                    print(f"Formatted {output_path}: {', '.join(changes)}")
            else:
                if not args.quiet:
                    output_path = (
                        Path(args.output) if args.output else xfail_path
                    )
                    print(f"No changes needed for {output_path}")

            return 0

    except Exception as e:
        print(f"Error formatting {xfail_path}: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage xfail lists for cuml.accel tests"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Edit tests command
    edit_parser = subparsers.add_parser(
        "edit-tests",
        help="Edit properties of a test and move it to a new group",
    )
    edit_parser.add_argument("xfail_list", help="Xfail list file to modify")
    edit_parser.add_argument(
        "tests",
        nargs="+",
        help="Test IDs to edit (use @filename to read from file)",
    )
    edit_parser.add_argument("--reason", help="New reason for the test")
    edit_parser.add_argument(
        "--strict", action="store_true", help="Make test strict"
    )
    edit_parser.add_argument(
        "--non-strict", action="store_true", help="Make test non-strict"
    )
    edit_parser.add_argument(
        "--run", action="store_true", help="Make test run"
    )
    edit_parser.add_argument(
        "--no-run", action="store_true", help="Make test not run"
    )
    edit_parser.add_argument("--condition", help="New condition for the test")
    edit_parser.add_argument("--marker", help="New marker for the test")
    edit_parser.add_argument(
        "--output", help="Output file (default: modify in-place)"
    )
    edit_parser.add_argument(
        "--in-place", action="store_true", help="Modify file in-place"
    )
    edit_parser.set_defaults(func=cmd_edit)

    # Add tests command
    add_tests_parser = subparsers.add_parser(
        "add-tests", help="Add tests to existing group"
    )
    add_tests_parser.add_argument(
        "xfail_list", help="Xfail list file to modify"
    )
    add_tests_parser.add_argument("--marker", help="Target group by marker")
    add_tests_parser.add_argument(
        "--reason-pattern", help="Target group by reason pattern"
    )
    add_tests_parser.add_argument(
        "tests",
        nargs="+",
        help="Test IDs to add (use @filename to read from file)",
    )
    add_tests_parser.add_argument(
        "--output", help="Output file (default: modify in-place)"
    )
    add_tests_parser.add_argument(
        "--in-place", action="store_true", help="Modify file in-place"
    )
    add_tests_parser.set_defaults(func=cmd_add_tests)

    # Remove tests command
    remove_tests_parser = subparsers.add_parser(
        "remove-tests", help="Remove tests from xfail list"
    )
    remove_tests_parser.add_argument(
        "xfail_list", help="Xfail list file to modify"
    )
    remove_tests_parser.add_argument(
        "tests",
        nargs="+",
        help="Test IDs to remove (use @filename to read from file)",
    )
    remove_tests_parser.add_argument(
        "--output", help="Output file (default: modify in-place)"
    )
    remove_tests_parser.add_argument(
        "--in-place", action="store_true", help="Modify file in-place"
    )
    remove_tests_parser.set_defaults(func=cmd_remove_tests)

    # Format command
    format_parser = subparsers.add_parser(
        "format",
        help="Auto-format xfail list(s) according to standards",
    )
    format_parser.add_argument(
        "xfail_list",
        nargs="+",
        help="Xfail list files to format",
    )
    format_parser.add_argument(
        "--output", help="Output file (default: modify in-place)"
    )
    format_parser.add_argument(
        "--cleanup", action="store_true", help="Remove empty groups"
    )
    format_parser.add_argument(
        "--check", action="store_true", help="Check if formatting needed"
    )
    format_parser.add_argument(
        "--ignore-validation",
        action="store_true",
        help="Ignore validation errors",
    )
    format_parser.add_argument(
        "--fix-validation", action="store_true", help="Fix validation errors"
    )
    format_parser.add_argument(
        "--quiet", action="store_true", help="Suppress output messages"
    )
    format_parser.set_defaults(func=cmd_format)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
