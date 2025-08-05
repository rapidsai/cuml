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
#
import cupy as cp
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from cudf import DataFrame
from distributed import Client

from cuml.dask.preprocessing import OrdinalEncoder


@pytest.mark.mg
def test_ordinal_encoder_df(client: Client) -> None:
    X = DataFrame({"cat": ["M", "F", "F"], "int": [1, 3, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)

    enc = OrdinalEncoder()
    enc.fit(X)
    Xt = enc.transform(X)

    X_1 = DataFrame({"cat": ["F", "F"], "int": [1, 2]})
    X_1 = dask_cudf.from_cudf(X_1, npartitions=2)

    enc = OrdinalEncoder(client=client)
    enc.fit(X)
    Xt_1 = enc.transform(X_1)

    Xt_r = Xt.compute()
    Xt_1_r = Xt_1.compute()
    assert Xt_1_r.iloc[0, 0] == Xt_r.iloc[1, 0]
    assert Xt_1_r.iloc[1, 0] == Xt_r.iloc[1, 0]
    assert Xt_1_r.iloc[0, 1] == Xt_r.iloc[0, 1]
    assert Xt_1_r.iloc[1, 1] == Xt_r.iloc[2, 1]

    # Turn Int64Index to RangeIndex for testing equality
    inv_Xt = enc.inverse_transform(Xt).compute().reset_index(drop=True)
    inv_Xt_1 = enc.inverse_transform(Xt_1).compute().reset_index(drop=True)

    X_r = X.compute()
    X_1_r = X_1.compute()

    assert inv_Xt.equals(X_r)
    assert inv_Xt_1.equals(X_1_r)

    assert enc.n_features_in_ == 2


@pytest.mark.mg
def test_ordinal_encoder_array(client: Client) -> None:
    X = DataFrame({"A": [4, 1, 1], "B": [1, 3, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2).values

    enc = OrdinalEncoder()
    enc.fit(X)
    Xt = enc.transform(X)

    X_1 = DataFrame({"A": [1, 1], "B": [1, 2]})
    X_1 = dask_cudf.from_cudf(X_1, npartitions=2).values

    enc = OrdinalEncoder(client=client)
    enc.fit(X)
    Xt_1 = enc.transform(X_1)

    Xt_r = Xt.compute()
    Xt_1_r = Xt_1.compute()
    assert Xt_1_r[0, 0] == Xt_r[1, 0]
    assert Xt_1_r[1, 0] == Xt_r[1, 0]
    assert Xt_1_r[0, 1] == Xt_r[0, 1]
    assert Xt_1_r[1, 1] == Xt_r[2, 1]

    inv_Xt = enc.inverse_transform(Xt)
    inv_Xt_1 = enc.inverse_transform(Xt_1)

    cp.testing.assert_allclose(X.compute(), inv_Xt.compute())
    cp.testing.assert_allclose(X_1.compute(), inv_Xt_1.compute())

    assert enc.n_features_in_ == 2


@pytest.mark.mg
@pytest.mark.parametrize("as_array", [True, False], ids=["cupy", "cudf"])
def test_handle_unknown(client, as_array: bool) -> None:
    X = DataFrame({"data": [0, 1]})
    Y = DataFrame({"data": [3, 1]})

    X = dask_cudf.from_cudf(X, npartitions=2)
    Y = dask_cudf.from_cudf(Y, npartitions=2)

    if as_array:
        X = X.values
        Y = Y.values

    enc = OrdinalEncoder(handle_unknown="error")
    enc = enc.fit(X)
    with pytest.raises(KeyError):
        enc.transform(Y).compute()

    enc = OrdinalEncoder(handle_unknown="ignore")
    enc = enc.fit(X)
    encoded = enc.transform(Y).compute()
    if as_array:
        np.isnan(encoded[0, 0])
    else:
        assert pd.isna(encoded.iloc[0, 0])
