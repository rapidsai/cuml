# Copyright (c) 2025, NVIDIA CORPORATION.
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

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from cuml.accel.core import ACCELERATED_MODULES
from cuml.accel.estimator_proxy import ProxyBase


def proxy_classes_with_set_output():
    for mod in ACCELERATED_MODULES:
        importlib.import_module(mod)

    return sorted(
        (
            cls
            for cls in ProxyBase.__subclasses__()
            if hasattr(cls, "set_output")
        ),
        key=lambda cls: cls.__name__,
    )


@pytest.mark.parametrize("cls", proxy_classes_with_set_output())
def test_set_output(cls):
    """Check that all proxy classes that define `set_output` handle it appropriately"""
    X, y = make_blobs(n_features=20, n_samples=100, random_state=42)

    model = cls().set_output(transform="pandas")
    if hasattr(model, "transform"):
        out = model.fit(X, y).transform(X)
    else:
        out = model.fit_transform(X, y)

    # Output is a pandas dataframe with non-default column names
    assert isinstance(out, pd.DataFrame)
    assert all(isinstance(c, str) for c in out.columns)

    # Can call `get_feature_names_out` without error
    names = model.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object

    # No host transfer required (this isn't strictly necessary, but is currently
    # true for all proxied estimators). Can revisit this check if it proves tricky
    # when adding new estimators.
    assert not hasattr(model._cpu, "n_features_in_")
