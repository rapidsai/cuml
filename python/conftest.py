#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import pytest

# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = ("cuml.test.plugins.quick_run_plugin")


def pytest_addoption(parser):
    # Any custom option, that should be available at any time (not just a
    # plugin), goes here.

    group = parser.getgroup('cuML Custom Options')

    group.addoption(
        "--run_stress",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'stress'. These are the most intense "
              "tests that take the longest to run and are designed to stress "
              "the hardware's compute resources."))

    group.addoption(
        "--run_quality",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'quality'. These tests are more "
              "computationally intense than 'unit', but less than 'stress'"))

    group.addoption(
        "--run_unit",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'unit'. These are the quickest tests "
              "that are focused on accuracy and correctness."))


def pytest_collection_modifyitems(config, items):

    should_run_quality = config.getoption("--run_quality")
    should_run_stress = config.getoption("--run_stress")

    # Run unit is implied if no --run_XXX is set
    should_run_unit = config.getoption("--run_unit") or not (
        should_run_quality or should_run_stress)

    # Mark the tests as "skip" if needed
    if not should_run_unit:
        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    if not should_run_quality:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if not should_run_stress:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
