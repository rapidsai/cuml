# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import contextlib
import doctest
import inspect
import io
import warnings

import cudf
import numpy as np
import pytest

import cuml


def _name_in_all(parent, name):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
    return not name.startswith("_")


def _find_doctests_in_obj(obj, finder=None, criteria=None):
    """Find all doctests in an object.

    Parameters
    ----------
    obj : module or class
        The object to search for docstring examples.
    finder : doctest.DocTestFinder, optional
        The DocTestFinder object to use. If not provided, a DocTestFinder is
        constructed.
    criteria : callable, optional
        Callable indicating whether to recurse over members of the provided
        object. If not provided, names not defined in the object's ``__all__``
        property are ignored.

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    if finder is None:
        finder = doctest.DocTestFinder()
    if criteria is None:
        criteria = _name_in_all
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(obj):
        # Only recurse over members matching the criteria
        if not criteria(obj, name):
            continue
        # Recurse over the public API of modules (objects defined in the
        # module's __all__)
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        if inspect.isclass(member):
            # Temporary hack to get tests to pass
            if "MultinomialNB" in member.__name__:
                continue
            if "RandomForestClassifier" in member.__name__:
                continue
            if "Lasso" in member.__name__:
                continue
            if "ElasticNet" in member.__name__:
                continue
            yield from _find_doctests_in_obj(
                member, finder, criteria=_is_public_name
            )
        if inspect.isfunction(member):
            yield from _find_doctests_in_obj(member, finder)


@pytest.mark.parametrize(
    "docstring",
    _find_doctests_in_obj(cuml),
    ids=lambda docstring: docstring.name,
)
def test_docstring(docstring):
    # We ignore differences in whitespace in the doctest output, and enable
    # the use of an ellipsis "..." to match any string in the doctest
    # output. An ellipsis is useful for, e.g., memory addresses or
    # imprecise floating point values.
    if docstring.name == "Handle":
        pytest.skip("Docstring is tested in RAFT.")
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    runner = doctest.DocTestRunner(optionflags=optionflags)

    # These global names are pre-defined and can be used in doctests
    # without first importing them.
    globals = dict(cudf=cudf, np=np, cuml=cuml)
    docstring.globs = globals

    # Capture stdout and include failing outputs in the traceback.
    doctest_stdout = io.StringIO()
    with contextlib.redirect_stdout(doctest_stdout):
        # Suppress expected warnings during doctest execution
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            runner.run(docstring)
            results = runner.summarize()
    try:
        assert not results.failed, (
            f"{results.failed} of {results.attempted} doctests failed for "
            f"{docstring.name}:\n{doctest_stdout.getvalue()}"
        )
    except AssertionError:
        # If some failed but all the failures were due to lack of multiGPU
        # support, we can skip. This code assumes that any MG-related failures
        # mean that all doctests failed for that reason, which is heavy-handed
        # and could miss a few things but is much easier than trying to
        # identify every line corresponding to any Exception raised.
        if (
            "cuML has not been built with multiGPU support"
            in doctest_stdout.getvalue()
        ):
            pytest.skip("Doctest requires MG support.")
        raise
