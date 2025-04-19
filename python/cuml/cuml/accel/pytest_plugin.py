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

from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import yaml
from packaging.requirements import Requirement

from cuml.accel.core import install


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
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
        return req.specifier.contains(installed_version)
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

    xfail_list_path = Path(xfail_list_path)
    if not xfail_list_path.exists():
        raise ValueError(f"Xfail list file not found: {xfail_list_path}")

    xfail_list = yaml.safe_load(xfail_list_path.read_text())

    if not isinstance(xfail_list, list):
        raise ValueError("Xfail list must be a list of test entries")

    # Convert list of dicts into dict mapping test IDs to lists of xfail configs
    xfail_configs = defaultdict(list)
    for entry in xfail_list:
        if not isinstance(entry, dict):
            raise ValueError("Xfail list entry must be a dictionary")
        if "id" not in entry:
            raise ValueError("Xfail list entry must contain an 'id' field")

        test_id = entry["id"]
        condition = True
        if "condition" in entry:
            condition = create_version_condition(entry["condition"])

        config = {
            "reason": entry.get("reason", "Test listed in xfail list"),
            "strict": entry.get("strict", True),
            "condition": condition,
        }

        xfail_configs[test_id].append(config)

    for item in items:
        test_id = f"{item.module.__name__}::{item.name}"
        if test_id in xfail_configs:
            for config in xfail_configs[test_id]:
                item.add_marker(
                    pytest.mark.xfail(
                        reason=config["reason"],
                        strict=config["strict"],
                        condition=config["condition"],
                    )
                )
