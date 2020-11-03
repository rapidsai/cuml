#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

import numbers
import os
import sys
import cupy as cp
import cupyx
import pytest
from pytest import Item
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# Stores incorrect uses of CumlArray on cuml.common.base.Base to print at the
# end
bad_cuml_array_loc = set()


def checked_isinstance(obj, class_name_dot_separated):
    """
    Small helper function to check instance of object that doesn't import
    class_path at import time, only at check time. Returns False if
    class_path cannot be imported.

    Parameters:
    -----------
    obj: Python object
        object to check if it is instance of a class
    class_name_dot_separated: list of str
        List of classes to check whether object is an instance of, each item
        can be a full dot  separated class like
        'cuml.dask.preprocessing.LabelEncoder'
    """
    ret = False
    for class_path in class_name_dot_separated:
        module_name, class_name = class_path.rsplit(".", 1)
        module = sys.modules[module_name]
        module_class = getattr(module, class_name, None)

        if module_class is not None:
            ret = isinstance(obj, module_class) or ret

    return ret


# Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is not used
def bad_allocator(nbytes):

    assert False, \
        "Using default cupy allocator instead of rmm.rmm_cupy_allocator"

    return None


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
def fail_on_bad_cuml_array_name(monkeypatch, request):

    if 'no_bad_cuml_array_check' in request.keywords:
        return

    from cuml.common import CumlArray
    from cuml.common.base import Base
    from cuml.common.input_utils import get_supported_input_type

    def patched__setattr__(self, name, value):

        if name == 'classes_' and \
                checked_isinstance(self,
                                   ['cuml.dask.preprocessing.LabelEncoder',
                                    'cuml.preprocessing.LabelEncoder']):
            # For label encoder, classes_ stores the set of unique classes
            # which is strings, and can't be saved as cuml array
            # even called `get_supported_input_type` causes a failure.
            pass
        else:
            supported_type = get_supported_input_type(value)

            if name == 'idf_':
                # We skip this test because idf_' for tfidf setter returns
                # a sparse diagonal matrix and getter gets a cupy array
                # see discussion at:
                # https://github.com/rapidsai/cuml/pull/2698/files#r471865982
                pass
            elif (supported_type == CumlArray):
                assert name.startswith("_"), \
                    ("Invalid CumlArray Use! CumlArray attributes need a "
                     "leading underscore. Attribute: '{}' "
                     "In: {}").format(name, self.__repr__())
            elif (supported_type == cp.ndarray
                  and cupyx.scipy.sparse.issparse(value)):
                # Leave sparse matrices alone for now.
                pass
            elif (supported_type is not None):
                if not isinstance(value, numbers.Number):
                    # Is this an estimated property?
                    # If so, should always be CumlArray
                    assert not name.endswith("_"), \
                        ("Invalid Estimated "
                         "Array-Like Attribute! Estimated attributes should "
                         "always be CumlArray. Attribute: '{}'"
                         " In: {}").format(name, self.__repr__())
                    assert not name.startswith("_"), \
                        ("Invalid Public "
                         "Array-Like Attribute! Public array-like attributes "
                         "should always be CumlArray. "
                         "Attribute: '{}' In: {}").format(name,
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
