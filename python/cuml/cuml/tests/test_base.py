# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import inspect

import cuml
import pytest
import numpydoc.docscrape

from pylibraft.common.cuda import Stream

from cuml.testing.utils import (
    get_classes_from_package,
    small_classification_dataset,
)
from cuml._thirdparty.sklearn.utils.skl_dependencies import (
    BaseEstimator as sklBaseEstimator,
)

all_base_children = get_classes_from_package(cuml, import_sub_packages=True)


def test_base_class_usage():
    # Ensure base class returns the 3 main properties needed by all classes
    base = cuml.Base()
    base.handle.sync()
    base_params = base._get_param_names()

    assert "handle" in base_params
    assert "verbose" in base_params
    assert "output_type" in base_params

    del base


def test_base_class_usage_with_handle():
    stream = Stream()
    handle = cuml.Handle(stream=stream)
    base = cuml.Base(handle=handle)
    base.handle.sync()
    del base


def test_base_hasattr():
    base = cuml.Base()
    # With __getattr__ overriding magic, hasattr should still return
    # True only for valid attributes
    assert hasattr(base, "handle")
    assert not hasattr(base, "somefakeattr")


@pytest.mark.parametrize("datatype", ["float32", "float64"])
@pytest.mark.parametrize("use_integer_n_features", [True, False])
def test_base_n_features_in(datatype, use_integer_n_features):
    X_train, _, _, _ = small_classification_dataset(datatype)
    integer_n_features = 8
    clf = cuml.Base()

    if use_integer_n_features:
        clf._set_n_features_in(integer_n_features)
        assert clf.n_features_in_ == integer_n_features
    else:
        clf._set_n_features_in(X_train)
        assert clf.n_features_in_ == X_train.shape[1]


@pytest.mark.parametrize("child_class", list(all_base_children.keys()))
def test_base_subclass_init_matches_docs(child_class: str):
    """
    This test is comparing the docstrings for arguments in __init__ for any
    class that derives from `Base`, We ensure that 1) the base arguments exist
    in the derived class, 2) The types and default values are the same and 3)
    That the docstring matches the base class

    This is to prevent multiple different docstrings for identical arguments
    throughout the documentation

    Parameters
    ----------
    child_class : str
        Classname to test in the dict all_base_children

    """
    klass = all_base_children[child_class]

    if issubclass(klass, sklBaseEstimator):
        pytest.skip(
            "Preprocessing models do not have "
            "the base arguments in constructors."
        )

    # To quickly find and replace all instances in the documentation, the below
    # regex's may be useful
    # output_type: r"^[ ]{4}output_type :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"
    # verbose: r"^[ ]{4}verbose :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"
    # handle: r"^[ ]{4}handle :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"

    def get_param_doc(param_doc_obj, name: str):
        found_doc = next((x for x in param_doc_obj if x.name == name), None)

        assert found_doc is not None, "Could not find {} in docstring".format(
            name
        )

        return found_doc

    # Load the base class signature, parse the docstring and pull out params
    base_sig = inspect.signature(cuml.Base, follow_wrapped=True)
    base_doc = numpydoc.docscrape.NumpyDocString(cuml.Base.__doc__)
    base_doc_params = base_doc["Parameters"]

    # Load the current class signature, parse the docstring and pull out params
    klass_sig = inspect.signature(klass, follow_wrapped=True)
    klass_doc = numpydoc.docscrape.NumpyDocString(klass.__doc__ or "")
    klass_doc_params = klass_doc["Parameters"]

    for name, param in base_sig.parameters.items():
        if param.name == "output_mem_type":
            continue  # TODO(wphicks): Add this to all algos
        # Ensure the base param exists in the derived
        assert param.name in klass_sig.parameters

        klass_param = klass_sig.parameters[param.name]

        # Ensure the default values are the same
        assert param.default == klass_param.default

        # Make sure we aren't accidentally a *args or **kwargs
        assert (
            klass_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or klass_param.kind == inspect.Parameter.KEYWORD_ONLY
        )

        if klass.__doc__ is not None:

            found_doc = get_param_doc(klass_doc_params, name)

            base_item_doc = get_param_doc(base_doc_params, name)

            # Ensure the docstring is identical
            assert (
                found_doc.type == base_item_doc.type
            ), "Docstring mismatch for {}".format(name)

            assert " ".join(found_doc.desc) == " ".join(base_item_doc.desc)


@pytest.mark.parametrize("child_class", list(all_base_children.keys()))
# ignore ColumnTransformer init warning
@pytest.mark.filterwarnings("ignore:Transformers are required")
def test_base_children__get_param_names(child_class: str):
    """
    This test ensures that the arguments in `Base.__init__` are available in
    all derived classes `_get_param_names`
    """

    klass = all_base_children[child_class]

    sig = inspect.signature(klass, follow_wrapped=True)

    try:
        bound = sig.bind()
        bound.apply_defaults()
    except TypeError:
        pytest.skip(
            "{}.__init__ requires non-default arguments to create. Skipping.".format(
                klass.__name__
            )
        )
    else:
        # Create an instance
        obj = klass(*bound.args, **bound.kwargs)

        param_names = obj._get_param_names()

        # Now ensure the base parameters are included in _get_param_names
        for name, param in sig.parameters.items():
            if param.name == "output_mem_type":
                continue  # TODO(wphicks): Add this to all algos
            if (
                param.kind == inspect.Parameter.VAR_KEYWORD
                or param.kind == inspect.Parameter.VAR_POSITIONAL
            ):
                continue

            assert name in param_names


# We explicitly skip the models in `cuml.tsa` since they match the statsmodels
# interface rather than the sklearn interface (https://github.com/rapidsai/cuml/issues/6258).
# Also skip a few classes that don't match this interface intentionally, since their sklearn
# equivalents are also exceptions.
@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in all_base_children.values()
        if not cls.__module__.startswith("cuml.tsa.")
        and cls
        not in {
            cuml.preprocessing.LabelBinarizer,
            cuml.preprocessing.LabelEncoder,
        }
    ],
)
def test_sklearn_methods_with_required_y_parameter(cls):
    optional_params = {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_KEYWORD,
    }
    for name in [
        "fit",
        "partial_fit",
        "score",
        "fit_transform",
        "fit_predict",
    ]:
        if (method := getattr(cls, name, None)) is None:
            # Method not defined, skip
            continue
        params = list(inspect.signature(method).parameters.values())
        # Assert method has a 2nd parameter named y, which is required by sklearn
        assert (
            len(params) > 2 and params[2].name == "y"
        ), f"`{name}` requires a `y` parameter, even if it's ignored"
        # Check that all remaining parameters are optional
        for param in params[3:]:
            assert (
                param.kind in optional_params
            ), f"`{name}` parameter `{param.name}` must be optional"
