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

This module provides tools for validating and auto-formatting xfail lists used in
cuml.accel testing. It offers both a programmatic API through the XfailManager
and XfailGroup classes, and a command-line interface for formatting operations.

Primary functions:
- Deterministic formatting and sorting of xfail lists
- Validation of group conditions
- Cleanup of empty groups

CLI Commands:
- format: Apply consistent formatting and sorting

The tool ensures xfail lists remain maintainable and produce clean diffs
in version control systems.
"""

import argparse
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

    def __lt__(self, other: "XfailGroup") -> bool:
        """Define ordering for deterministic sorting of groups."""
        if not isinstance(other, XfailGroup):
            return NotImplemented

        # Primary sort: by marker (None values last)
        self_marker = self.marker or "zzz_no_marker"
        other_marker = other.marker or "zzz_no_marker"
        if self_marker != other_marker:
            return self_marker < other_marker

        # Secondary sort: by reason (alphabetically)
        if self.reason != other.reason:
            return self.reason < other.reason

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

    def save(self, xfail_list_path: Union[str, Path]) -> None:
        """Save xfail list to YAML file."""
        path = Path(xfail_list_path)

        # Merge groups with identical properties before saving
        self._merge_identical_groups()

        # Filter out empty groups and sort deterministically
        non_empty_groups = [group for group in self.groups if group]
        sorted_groups = sorted(non_empty_groups)

        data = [group.to_dict() for group in sorted_groups]

        # Setup YAML formatting to match summarize-results.py
        setup_yaml()
        with path.open("w") as f:
            yaml.dump(data, f, sort_keys=False, width=float("inf"))

    def cleanup_empty_groups(self) -> int:
        """Remove groups with no tests.

        Returns:
            Number of groups removed
        """
        empty_groups = [group for group in self.groups if not group]
        for group in empty_groups:
            self.groups.remove(group)
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

    def _merge_identical_groups(self) -> int:
        """Merge groups that have identical properties except for tests.

        Returns:
            Number of groups that were merged (removed)
        """
        # Group the groups by their properties (excluding tests)
        property_groups = defaultdict(list)

        for group in self.groups:
            # Create a key based on all properties except tests
            key = (
                group.reason,
                group.strict,
                group.condition,
                group.run,
                group.marker,
            )
            property_groups[key].append(group)

        merged_count = 0
        new_groups = []

        for groups_with_same_properties in property_groups.values():
            if len(groups_with_same_properties) == 1:
                # Only one group with these properties, keep as-is
                new_groups.append(groups_with_same_properties[0])
            else:
                # Multiple groups with same properties, merge them
                merged_count += len(groups_with_same_properties) - 1

                # Take the first group as the base
                base_group = groups_with_same_properties[0]

                # Collect all unique tests from all groups
                all_tests = set(base_group.tests)
                for group in groups_with_same_properties[1:]:
                    all_tests.update(group.tests)

                # Update the base group with all tests
                base_group.tests = sorted(list(all_tests))
                new_groups.append(base_group)

        # Replace the groups list with merged groups
        self.groups = new_groups

        return merged_count


# CLI Commands Implementation


def cmd_format(args):
    """Auto-format xfail list(s) according to standards."""

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
    results = []

    for xfail_path in xfail_paths:
        results.append(_format_single_file(xfail_path, args))

    return 1 if any(results) else 0


def _format_single_file(xfail_path, args):
    """Format a single xfail file.

    Returns 0 on success, 1 on error/changes needed.
    """

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
        if validation_errors:
            print(f"Validation errors in {xfail_path}:", file=sys.stderr)
            for error in validation_errors:
                print(f"  {error}", file=sys.stderr)
            return 1

        # Save changes
        manager.save(xfail_path)
        formatted_content = xfail_path.read_text()

        # Check if changes were made
        content_changed = original_content != formatted_content

        # Report changes
        changes = []
        if content_changed:
            changes.append("formatting applied")
        if removed_groups > 0:
            changes.append(f"{removed_groups} empty groups removed")

        if changes:
            print(f"Formatted {xfail_path}: {', '.join(changes)}")
        else:
            print(f"No changes needed for {xfail_path}")

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
        "--cleanup", action="store_true", help="Remove empty groups"
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
