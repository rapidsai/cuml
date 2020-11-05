import pytest

pytest_plugins = ("cuml.test.plugins.profiling_plugin",
                  "cuml.test.plugins.quick_run_plugin")


def pytest_addoption(parser):
    parser.addoption("--run_stress",
                     action="store_true",
                     default=False,
                     help="run stress tests")

    parser.addoption("--run_quality",
                     action="store_true",
                     default=False,
                     help="run quality tests")

    parser.addoption("--run_unit",
                     action="store_true",
                     default=False,
                     help="run unit tests")

    parser.addoption(
        "--check_memory",
        action="store_true",
        default=False,
        help=("Adds a memory checker plugin that reports tests with memory "
              "leaks"))


def pytest_configure(config):
    # Import the check memory plugin if specified (better than always importing
    # it)
    use_check_memory = config.getoption("--check_memory")

    plugin_name = "cuml.test.plugins.check_memory_plugin"

    if use_check_memory and not config.pluginmanager.has_plugin(plugin_name):
        config.pluginmanager.import_plugin(plugin_name)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_quality"):
        # --run_quality given in cli: do not skip quality tests
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    else:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests

        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    else:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
