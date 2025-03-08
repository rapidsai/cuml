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
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder as skOrdinalEncoder

from cuml.internals.safe_imports import gpu_only_import_from
from cuml.preprocessing import OrdinalEncoder

DataFrame = gpu_only_import_from("cudf", "DataFrame")


@pytest.fixture
def test_sample():
    X = DataFrame({"cat": ["M", "F", "F"], "num": [1, 3, 2]})
    return X


def test_ordinal_encoder_df(test_sample) -> None:
    X = test_sample
    enc = OrdinalEncoder()
    enc.fit(X)
    Xt = enc.transform(X)

    X_1 = DataFrame({"cat": ["F", "F"], "num": [1, 2]})
    Xt_1 = enc.transform(X_1)

    assert Xt_1.iloc[0, 0] == Xt.iloc[1, 0]
    assert Xt_1.iloc[1, 0] == Xt.iloc[1, 0]
    assert Xt_1.iloc[0, 1] == Xt.iloc[0, 1]
    assert Xt_1.iloc[1, 1] == Xt.iloc[2, 1]

    inv_Xt = enc.inverse_transform(Xt)
    inv_Xt_1 = enc.inverse_transform(Xt_1)

    assert inv_Xt.equals(X)
    assert inv_Xt_1.equals(X_1)

    assert enc.n_features_in_ == 2


def test_ordinal_encoder_array() -> None:
    X = DataFrame({"A": [4, 1, 1], "B": [1, 3, 2]}).values
    enc = OrdinalEncoder()
    enc.fit(X)
    Xt = enc.transform(X)

    X_1 = DataFrame({"A": [1, 1], "B": [1, 2]}).values
    Xt_1 = enc.transform(X_1)

    assert Xt_1[0, 0] == Xt[1, 0]
    assert Xt_1[1, 0] == Xt[1, 0]
    assert Xt_1[0, 1] == Xt[0, 1]
    assert Xt_1[1, 1] == Xt[2, 1]

    inv_Xt = enc.inverse_transform(Xt)
    inv_Xt_1 = enc.inverse_transform(Xt_1)

    cp.testing.assert_allclose(X, inv_Xt)
    cp.testing.assert_allclose(X_1, inv_Xt_1)

    assert enc.n_features_in_ == 2


def test_ordinal_array() -> None:
    X = cp.arange(32).reshape(16, 2)

    enc = OrdinalEncoder()
    enc.fit(X)
    Xt = enc.transform(X)

    Xh = cp.asnumpy(X)
    skenc = skOrdinalEncoder()
    skenc.fit(Xh)
    Xt_sk = skenc.transform(Xh)

    cp.testing.assert_allclose(Xt, Xt_sk)


def test_output_type(test_sample) -> None:
    X = test_sample
    enc = OrdinalEncoder(output_type="cupy").fit(X)
    assert isinstance(enc.transform(X), cp.ndarray)
    enc = OrdinalEncoder(output_type="cudf").fit(X)
    assert isinstance(enc.transform(X), DataFrame)
    enc = OrdinalEncoder(output_type="pandas").fit(X)
    assert isinstance(enc.transform(X), pd.DataFrame)
    enc = OrdinalEncoder(output_type="numpy").fit(X)
    assert isinstance(enc.transform(X), np.ndarray)
    # output_type == "input"
    enc = OrdinalEncoder().fit(X)
    assert isinstance(enc.transform(X), DataFrame)


def test_feature_names(test_sample) -> None:
    enc = OrdinalEncoder().fit(test_sample)
    assert enc.feature_names_in_ == ["cat", "num"]


@pytest.mark.parametrize("as_array", [True, False], ids=["cupy", "cudf"])
def test_handle_unknown(as_array: bool) -> None:
    X = DataFrame({"data": [0, 1]})
    Y = DataFrame({"data": [3, 1]})

    if as_array:
        X = X.values
        Y = Y.values

    enc = OrdinalEncoder(handle_unknown="error")
    enc = enc.fit(X)
    with pytest.raises(KeyError):
        enc.transform(Y)

    enc = OrdinalEncoder(handle_unknown="ignore")
    enc = enc.fit(X)
    encoded = enc.transform(Y)
    if as_array:
        np.isnan(encoded[0, 0])
    else:
        assert pd.isna(encoded.iloc[0, 0])
