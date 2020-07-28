import os

import cupy as cp
import pytest
from pytest import Item
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Stores incorrect uses of CumlArray on cuml.common.base.Base to print at the
# end
bad_cuml_array_loc = set()


def pytest_configure(config):
    cp.cuda.set_allocator(None)


# Use the runtest_makereport hook to get the result of the test. This is
# necessary because pytest has some magic to extract the Cython source file
# from the traceback
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call):

    # Yield to the default implementation and get the result
    outcome = yield
    report = outcome.get_result()

    # report = TestReport.from_item_and_call(item, call)

    if (report.failed):
        # Ensure these attributes exist. They can be missing if something else
        # failed outside of the test
        if (hasattr(report.longrepr, "reprtraceback") and
                hasattr(report.longrepr.reprtraceback, "reprentries")):

            for entry in reversed(report.longrepr.reprtraceback.reprentries):

                if (os.path.splitext(entry.reprfileloc.path)[1] == ".pyx"):

                    true_path = "{}:{}".format(entry.reprfileloc.path,
                                               entry.reprfileloc.lineno)

                    bad_cuml_array_loc.add((true_path,
                                            entry.reprfileloc.message))

                    break


# Closing hook to display the file/line numbers at the end of the test
def pytest_unconfigure(config):

    if (len(bad_cuml_array_loc) > 0):

        print("Bad uses of CumlArray on Base:")

        prefix = ""

        # Depending on where pytest was launched from, it may need to append
        # "python"
        if (not os.path.basename(os.path.abspath(os.curdir))
                .endswith("python")):
            prefix = "python"

        for location, message in bad_cuml_array_loc:
            print("{} {}".format(os.path.abspath(
                os.path.join(prefix, location)), message))


# This fixture will monkeypatch cuml.common.base.Base to check for incorrect
# uses of CumlArray.
@pytest.fixture(autouse=True)
def fail_on_bad_cuml_array_name(monkeypatch):

    from cuml.common import CumlArray
    from cuml.common.base import Base
    from cuml.common.array import CumlArrayDescriptor
    from cuml.common.input_utils import get_supported_input_type

    def patched__setattr__(self, name, value):

        supported_type = get_supported_input_type(value)

        curr_allocator = cp.cuda.get_allocator()

        if (supported_type == CumlArray):
            if (name in type(self).__dict__ and type(type(self).__dict__[name]) == CumlArrayDescriptor):
                # This situation is OK if we are using a descriptor
                pass
            else:
                assert name.startswith("_"), "Invalid CumlArray Use! Attribute: \
                    '{}' In: {}".format(name, self.__repr__())
        elif (supported_type is not None):
            # Additional checks for settings specific types on an object
            assert not name.endswith("_") and not name.startswith("_"), "Attribute: '{}' In: {}".format(name, self.__repr__())
            # print("Setting non-CumlArray type. Class: {}, Attr: {}, Value Type: {}".format(self.__class__.__name__, name, str(supported_type)))

        return super(Base, self).__setattr__(name, value)

    """Monkeypatch CumlArray.__setattr__ to assert array attributes have a
       leading underscore. i.e. `self._my_variable_ = CumlArray.ones(10)`."""
    monkeypatch.setattr(Base, "__setattr__", patched__setattr__)


@pytest.fixture(scope="module")
def nlp_20news():
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


def pytest_addoption(parser):
    parser.addoption("--run_stress", action="store_true",
                     default=False, help="run stress tests")

    parser.addoption("--run_quality", action="store_true",
                     default=False, help="run quality tests")

    parser.addoption("--run_unit", action="store_true",
                     default=False, help="run unit tests")


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

        return

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

        return

    else:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
