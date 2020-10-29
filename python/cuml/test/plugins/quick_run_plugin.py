import typing

import _pytest.python


def pytest_addoption(parser):

    parser.addoption("--quick_run",
                     default=None,
                     type=int,
                     help="run unit tests")


def pytest_collection_modifyitems(config, items):

    quick_run = config.getoption("--quick_run")

    if (quick_run):
        root_node = {}
        leafs = []

        def get_leaf(node_list: list) -> list:

            curr_node = root_node

            for n in node_list:
                name = getattr(n, "originalname", n.name)

                if (name not in curr_node):
                    if (isinstance(n, _pytest.python.Function)):
                        curr_node[name] = []
                        leafs.append(curr_node[name])
                    else:
                        curr_node[name] = {}

                curr_node = curr_node[name]

            return curr_node

        for item in items:
            leaf = get_leaf(item.listchain())

            leaf.append(item)

        selected_items = []
        deselected_items = []

        def process_leaf_seeonce(leaf: typing.List[_pytest.python.Function]):
            seen = {}

            def has_been_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if (key not in seen):
                        return False

                    if (val not in seen[key]):
                        return False

                return True

            def update_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if (key not in seen):
                        seen[key] = []

                    if (val not in seen[key]):
                        seen[key].append(val)

            for f in leaf:
                # If no callspec, this is the only function call
                if (not hasattr(f, "callspec")):
                    selected_items.append(f)
                    continue

                callspec = f.callspec

                if (has_been_seen(callspec)):
                    deselected_items.append(f)
                else:
                    # Add to seen and selected
                    selected_items.append(f)

                    update_seen(callspec)

        for leaf in leafs:
            process_leaf_seeonce(leaf)

        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
