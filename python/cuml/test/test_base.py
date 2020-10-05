# Copyright (c) 2019, NVIDIA CORPORATION.
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
from cuml.test.utils import (get_classes_from_package,
                             small_classification_dataset)

all_base_children = get_classes_from_package(cuml, import_sub_packages=True)


def test_base_class_usage():
    base = cuml.Base()
    base.handle.sync()
    base_params = base.get_param_names()
    assert base_params == []
    del base


def test_base_class_usage_with_handle():
    handle = cuml.Handle()
    stream = cuml.cuda.Stream()
    handle.setStream(stream)
    base = cuml.Base(handle=handle)
    base.handle.sync()
    del base


def test_base_hasattr():
    base = cuml.Base()
    # With __getattr__ overriding magic, hasattr should still return
    # True only for valid attributes
    assert hasattr(base, "handle")
    assert not hasattr(base, "somefakeattr")


@pytest.mark.parametrize('datatype', ["float32", "float64"])
@pytest.mark.parametrize('use_integer_n_features', [True, False])
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


@pytest.mark.parametrize('child_class', list(all_base_children.keys()))
def test_base_children_init(child_class: str):

    # Regex for find and replace:
    # output_type: `^[ ]{4}output_type :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+`

    def get_param_doc(param_doc_obj, name: str):
        found_doc = next((x for x in param_doc_obj if x.name == name), None)

        assert found_doc is not None, \
            "Could not find {} in docstring".format(name)

        return found_doc

    base_sig = inspect.signature(cuml.Base, follow_wrapped=True)

    base_doc = numpydoc.docscrape.NumpyDocString(cuml.Base.__doc__)

    base_doc_params = base_doc["Parameters"]

    klass = all_base_children[child_class]

    klass_sig = inspect.signature(klass, follow_wrapped=True)

    klass_doc = numpydoc.docscrape.NumpyDocString(klass.__doc__ or "")

    klass_doc_params = klass_doc["Parameters"]

    for name, param in base_sig.parameters.items():
        assert param.name in klass_sig.parameters

        klass_param = klass_sig.parameters[param.name]

        assert param.default == klass_param.default

        assert (klass_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                or klass_param.kind == inspect.Parameter.KEYWORD_ONLY)

        if (klass.__doc__ is not None):

            found_doc = get_param_doc(klass_doc_params, name)

            base_item_doc = get_param_doc(base_doc_params, name)

            assert found_doc.type == base_item_doc.type, \
                "Docstring mismatch for {}".format(name)

            assert " ".join(found_doc.desc) == " ".join(base_item_doc.desc)


@pytest.mark.parametrize('child_class', list(all_base_children.keys()))
def test_base_children_get_param_names(child_class: str):

    klass = all_base_children[child_class]

    sig = inspect.signature(klass, follow_wrapped=True)

    try:
        bound = sig.bind()
        bound.apply_defaults()
    except TypeError:
        pytest.skip(
            "{}.__init__ requires non-default arguments to create. Skipping.".
            format(klass.__name__))
    else:
        # Create an insteance
        obj = klass(*bound.args, **bound.kwargs)

        param_names = obj.get_param_names()

        for name, param in sig.parameters.items():

            if (param.kind == inspect.Parameter.VAR_KEYWORD
                    or param.kind == inspect.Parameter.VAR_POSITIONAL):
                continue

            assert name in param_names
