#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import typing

import _pytest.python
import pytest


def pytest_addoption(parser):

    group = parser.getgroup("Quick Run Plugin")

    group.addoption(
        "--quick_run",
        default=False,
        action="store_true",
        help=(
            "Selecting this option will reduce the number of "
            "tests run by only running parameter combinations "
            "if one of the parameters has never been seen "
            "before. Useful for testing code correctness while "
            "not running all numeric tests for the algorithms."
        ),
    )


# This hook must be run last after all others as some plugins may have skipped
# some tests
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):

    quick_run = config.getoption("--quick_run")

    if quick_run:
        root_node = {}
        leafs = []

        def get_leaf(node_list: list) -> list:
            """
            Responsible for returning the leaf test node and building any
            interior nodes in the process

            Parameters
            ----------
            node_list : list List of strings for each pytest node returned from
                `listchain()`

            Returns
            -------
            list Returns the leaf node containing a list of all tests in the
                leaf
            """

            curr_node = root_node

            for n in node_list:
                name = getattr(n, "originalname", n.name)

                # Add the interior node if it doesn't exist. Must be a function
                # to be a leaf
                if name not in curr_node:
                    if isinstance(n, _pytest.python.Function):
                        curr_node[name] = []
                        leafs.append(curr_node[name])
                    else:
                        curr_node[name] = {}

                curr_node = curr_node[name]

            return curr_node

        # Loop over all nodes and generate a tree structure from their layout
        for item in items:
            leaf = get_leaf(item.listchain())

            leaf.append(item)

        selected_items = []
        deselected_items = []

        def process_leaf_seeonce(leaf: typing.List[_pytest.python.Function]):
            seen = {}

            # Returns True if this test's parameters are too similar to another
            # test already in the selected list
            def has_been_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if key not in seen:
                        return False

                    if val not in seen[key]:
                        return False

                return True

            # Updates the seen dictionary with the parameters from this test
            def update_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if key not in seen:
                        seen[key] = []

                    if val not in seen[key]:
                        seen[key].append(val)

            for f in leaf:

                # If this is going to be skipped, add to deselected. No need to
                # run it
                if f.get_closest_marker("skip") is not None:
                    deselected_items.append(f)
                    continue

                # If no callspec, this is the only function call. Must be run
                if not hasattr(f, "callspec"):
                    selected_items.append(f)
                    continue

                callspec = f.callspec

                # Check if this has been seen
                if has_been_seen(callspec):
                    deselected_items.append(f)
                else:
                    # Otherwise, add to seen and selected
                    selected_items.append(f)

                    update_seen(callspec)

        # Now looping over all leafs, see which ones we can process only once
        for leaf in leafs:
            process_leaf_seeonce(leaf)

        # Deselect the skipped nodes and shorten the items list
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
