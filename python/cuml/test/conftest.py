import os

import cupy as cp
import cupyx
import pytest
import _pytest.config
import _pytest.terminal
from pytest import Item
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numbers

import rmm

# rmm.reinitialize(logging=True, log_file_name="test_log.txt")

# Stores incorrect uses of CumlArray on cuml.common.base.Base to print at the
# end
bad_cuml_array_loc = set()


# Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is not used
def bad_allocator(nbytes):

    assert False, "Using default cupy allocator instead of rmm.rmm_cupy_allocator"

    return None



saved_allocator = rmm.rmm_cupy_allocator

def counting_rmm_allocator(nbytes):

    import cuml.common.array

    cuml.common.array._increment_malloc(nbytes)

    # if (global_output_type_data.root_cm is not None):

    #     current_func = global_output_type_data.root_cm.get_current_func()

    #     if (current_func):
    #         print("{} Allocating {} bytes from {}:{}".format(repr(current_func), nbytes, current_func.func_code.co_filename, current_func.func_code.co_firstlineno))

    return saved_allocator(nbytes)

rmm.rmm_cupy_allocator = counting_rmm_allocator

def pytest_configure(config):
    cp.cuda.set_allocator(counting_rmm_allocator)

@pytest.fixture(scope="function")
def cupy_allocator_fixture():

    # Disable creating cupy arrays
    cp.cuda.set_allocator(bad_allocator)

    yield

    # Reset creating cupy arrays
    cp.cuda.set_allocator(None)


# Use the runtest_makereport hook to get the result of the test. This is
# necessary because pytest has some magic to extract the Cython source file
# from the traceback
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call):

    # Yield to the default implementation and get the result
    outcome = yield
    report = outcome.get_result()

    if (report.failed):

        # Save the abs path to this file. We will only mark bad CumlArray uses
        # if the assertion failure comes from this file
        conf_test_path = os.path.abspath(__file__)

        found_assert = False

        # Ensure these attributes exist. They can be missing if something else
        # failed outside of the test
        if (hasattr(report.longrepr, "reprtraceback")
                and hasattr(report.longrepr.reprtraceback, "reprentries")):

            for entry in reversed(report.longrepr.reprtraceback.reprentries):

                if (not found_assert and
                        entry.reprfileloc.message.startswith("AssertionError")
                        and os.path.abspath(
                            entry.reprfileloc.path) == conf_test_path):
                    found_assert = True
                elif (found_assert):
                    true_path = "{}:{}".format(entry.reprfileloc.path,
                                               entry.reprfileloc.lineno)

                    bad_cuml_array_loc.add(
                        (true_path, entry.reprfileloc.message))

                    break


def pytest_terminal_summary(terminalreporter: _pytest.terminal.TerminalReporter, exitstatus: pytest.ExitCode, config: _pytest.config.Config):

    terminalreporter.write_sep("=", "CumlArray Summary")

    import cuml.common.array

    terminalreporter.write_line("To Output Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._to_output_counts))

    terminalreporter.write_line("From Array Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._from_array_counts))

    terminalreporter.write_line("RMM Malloc: Count={}, Size={}".format(cuml.common.array._malloc_count.get(), cuml.common.array._malloc_nbytes.get()))


# Closing hook to display the file/line numbers at the end of the test
def pytest_unconfigure(config):

    def split_exists(filename: str) -> bool:
        strip_colon = filename[:filename.rfind(":")]
        return os.path.exists(strip_colon)

    if (len(bad_cuml_array_loc) > 0):

        print("Incorrect CumlArray uses in class derived from "
              "cuml.common.base.Base:")

        prefix = ""

        # Depending on where pytest was launched from, it may need to append
        # "python"
        if (not os.path.basename(os.path.abspath(
                os.curdir)).endswith("python")):
            prefix = "python"

        for location, message in bad_cuml_array_loc:

            combined_path = os.path.abspath(location)

            # Try appending prefix if that file doesnt exist
            if (not split_exists(combined_path)):
                combined_path = os.path.abspath(os.path.join(prefix, location))

                # If that still doesnt exist, just use the original
                if (not split_exists(combined_path)):
                    combined_path = location

            print("{} {}".format(combined_path, message))

        print(
            "See https://github.com/rapidsai/cuml/issues/2456#issuecomment-666106406"  # noqa
            " for more information on naming conventions")


# This fixture will monkeypatch cuml.common.base.Base to check for incorrect
# uses of CumlArray.
@pytest.fixture(autouse=True)
def fail_on_old_cuml_array_conversion(monkeypatch):

    from cuml.common import CumlArray
    from cuml.common.base import Base

    saved_get_attr = Base.__getattr__

    def patched__getattr__(self, name):

        real_name = '_' + name

        if real_name in self.__dict__.keys():

            assert not isinstance(self.__dict__[real_name], CumlArray), \
                "Old-style CumlArray conversion. Use CumlArrayDescriptor"

        return saved_get_attr(self, name)

    # Monkeypatch CumlArray.__setattr__ to test for incorrect uses of
    # array-like objects
    monkeypatch.setattr(Base, "__getattr__", patched__getattr__)


# This fixture will monkeypatch cuml.common.base.Base to check for incorrect
# uses of CumlArray.
@pytest.fixture(autouse=True)
def fail_on_bad_cuml_array_name(monkeypatch, request):

    if 'no_bad_cuml_array_check' in request.keywords:
        return

    from cuml.common import CumlArray
    from cuml.common.base import Base
    from cuml.common.input_utils import get_supported_input_type

    def patched__setattr__(self, name, value):

        supported_type = get_supported_input_type(value)

        if name == 'idf_':
            # We skip this test because idf_' for tfidf setter returns
            # a sparse diagonal matrix and getter gets a cupy array
            # see discussion at:
            # https://github.com/rapidsai/cuml/pull/2698/files#r471865982
            pass
        elif (supported_type == CumlArray):
            assert name.startswith("_"), "Invalid CumlArray Use! CumlArray \
                attributes need a leading underscore. Attribute: '{}' In: {}" \
                    .format(name, self.__repr__())
        elif (supported_type == cp.ndarray
              and cupyx.scipy.sparse.issparse(value)):
            # Leave sparse matrices alone for now.
            pass
        elif (supported_type is not None):
            if not isinstance(value, numbers.Number):
                # Is this an estimated property?
                # If so, should always be CumlArray
                assert not name.endswith("_"), "Invalid Estimated Array-Like \
                    Attribute! Estimated attributes should always be \
                    CumlArray. \
                    Attribute: '{}' In: {}".format(name, self.__repr__())
                assert not name.startswith("_"), "Invalid Public Array-Like \
                    Attribute! Public array-like attributes should always be \
                    CumlArray. Attribute: '{}' In: {}".format(name,
                                                              self.__repr__())
            else:
                # Estimated properties can be numbers
                pass

        return super(Base, self).__setattr__(name, value)

    # Monkeypatch CumlArray.__setattr__ to test for incorrect uses of
    # array-like objects
    # monkeypatch.setattr(Base, "__setattr__", patched__setattr__)


@pytest.fixture(scope="module")
def nlp_20news():
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


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
