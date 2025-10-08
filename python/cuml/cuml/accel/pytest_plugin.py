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

import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

from cuml.accel._sklearn_patch import apply_sklearn_patches
from cuml.accel.core import install


class UnmatchedXfailTests(UserWarning):
    """Warning raised when xfail entries in the configuration file don't match any actual tests.

    This warning is raised during pytest collection when there are entries in the xfail
    list that don't correspond to any existing test functions. This typically indicates
    either:
    1. Tests have been renamed or removed but the xfail list wasn't updated
    2. There are typos in the test IDs in the xfail list
    3. The tests only exist for specific versions of dependencies (check the condition
       field in the xfail list)
    """

    ...


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests

    # Apply sklearn patches BEFORE installing cuml.accel to prevent duplicates
    apply_sklearn_patches()

    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded sklearn. Interposing failed."
        )


def pytest_addoption(parser):
    """Add command line option for xfail list file."""
    parser.addoption(
        "--xfail-list",
        action="store",
        help="Path to YAML file containing list of test IDs to mark as xfail",
    )


def create_version_condition(condition_str: str) -> bool:
    """Evaluate a version condition immediately.

    Args:
        condition_str: String in format 'package[comparison]version'
                      For example:
                      - 'scikit-learn>=1.5.2'
                      - 'numpy<2.0.0'
                      - 'pandas==2.1.0'

    Returns:
        bool: True if the condition is met, False otherwise
    """
    if not condition_str:
        return True

    try:
        req = Requirement(condition_str)
        installed_version = version(req.name)
        return req.specifier.contains(installed_version, prereleases=True)
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Apply xfail markers to tests listed in the xfail list file."""
    # Import pytest lazily to avoid requiring it for normal cuml usage.
    # pytest is only needed when running tests.
    import pytest

    xfail_list_path = config.getoption("xfail_list")
    if not xfail_list_path:
        return

    import yaml  # needed to parse the YAML-formatted xfail list

    xfail_list_path = Path(xfail_list_path)
    if not xfail_list_path.exists():
        raise ValueError(f"Xfail list file not found: {xfail_list_path}")

    xfail_list = yaml.safe_load(xfail_list_path.read_text())

    if not isinstance(xfail_list, list):
        raise ValueError("Xfail list must be a list of test groups")

    # Create markers for all unique markers in the xfail list
    markers = {
        marker: pytest.mark.__getattr__(marker)
        for group in xfail_list
        if (marker := group.get("marker"))
    }
    # Convert list of groups into dict mapping test IDs to lists of xfail
    # configs
    xfail_configs = defaultdict(list)
    for group in xfail_list:
        if not isinstance(group, dict):
            raise ValueError("Xfail list entry must be a dictionary")
        if "reason" not in group:
            raise ValueError("Xfail list entry must contain a 'reason' field")
        if "tests" not in group:
            raise ValueError("Xfail list entry must contain a 'tests' field")

        reason = group["reason"]
        strict = group.get("strict", True)
        run = group.get("run", True)
        tests = group["tests"]
        condition = True
        if "condition" in group:
            condition = create_version_condition(group["condition"])
        marker = markers.get(group.get("marker", None), None)

        config = {
            "reason": reason,
            "strict": strict,
            "run": run,
            "condition": condition,
            "extra_marker": marker,
        }

        for test_id in tests:
            xfail_configs[test_id].append(config)

    # Track which xfail test IDs were actually found
    found_xfail_tests = set()

    for item in items:
        test_id = f"{item.module.__name__}::{item.name}"
        if test_id in xfail_configs:
            found_xfail_tests.add(test_id)
            for config in xfail_configs[test_id]:
                # Add the xfail marker
                item.add_marker(
                    pytest.mark.xfail(
                        reason=config["reason"],
                        strict=config["strict"],
                        run=config["run"],
                        condition=config["condition"],
                    )
                )
                # If there's a marker, add it as a proper pytest marker
                if extra_marker := config["extra_marker"]:
                    item.add_marker(extra_marker)

    # Check for xfail entries that don't match any actual tests
    # Only include tests where at least one config has a met condition
    expected_tests = {
        test_id
        for test_id, configs in xfail_configs.items()
        if any(config["condition"] for config in configs)
    }
    missing_tests = expected_tests - found_xfail_tests
    if missing_tests:
        missing_list = sorted(missing_tests)
        print("Did not find the following test ids:")
        print("\n".join(missing_list))

        warnings.warn(
            f"Found {len(missing_list)} xfail entries that don't match any present tests",
            category=UnmatchedXfailTests,
        )
