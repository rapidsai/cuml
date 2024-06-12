# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
from cuml.internals.safe_imports import cpu_only_import_from
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from cuml.testing.utils import (
    stress_param,
    generate_inputs_from_categories,
    assert_inverse_equal,
    from_df_to_numpy,
)
from cuml.dask.preprocessing import OneHotEncoder
import dask.array as da
from cuml.internals.safe_imports import cpu_only_import
from cudf import DataFrame, Series
import pytest
from cuml.internals.safe_imports import gpu_only_import

dask_cudf = gpu_only_import("dask_cudf")
cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")
assert_frame_equal = cpu_only_import_from(
    "pandas.testing", "assert_frame_equal"
)


@pytest.mark.mg
def test_onehot_vs_skonehot(client):
    X = DataFrame({"gender": ["Male", "Female", "Female"], "int": [1, 3, 2]})
    skX = from_df_to_numpy(X)
    X = dask_cudf.from_cudf(X, npartitions=2)

    enc = OneHotEncoder(sparse_output=False)
    skohe = SkOneHotEncoder(sparse_output=False)

    ohe = enc.fit_transform(X)
    ref = skohe.fit_transform(skX)

    cp.testing.assert_array_equal(ohe.compute(), ref)


@pytest.mark.mg
@pytest.mark.parametrize(
    "drop", [None, "first", {"g": Series("F"), "i": Series(3)}]
)
def test_onehot_inverse_transform(client, drop):
    df = DataFrame({"g": ["M", "F", "F"], "i": [1, 3, 2]})
    X = dask_cudf.from_cudf(df, npartitions=2)

    enc = OneHotEncoder(drop=drop)
    ohe = enc.fit_transform(X)
    inv = enc.inverse_transform(ohe)
    assert_frame_equal(
        inv.compute().to_pandas().reset_index(drop=True), df.to_pandas()
    )


@pytest.mark.mg
def test_onehot_categories(client):
    X = DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)
    cats = DataFrame({"chars": ["a", "b", "c"], "int": [0, 1, 2]})
    enc = OneHotEncoder(categories=cats, sparse_output=False)
    ref = cp.array(
        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    )
    res = enc.fit_transform(X)
    cp.testing.assert_array_equal(res.compute(), ref)


@pytest.mark.mg
def test_onehot_fit_handle_unknown(client):
    X = DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    Y = DataFrame({"chars": ["c", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)

    enc = OneHotEncoder(handle_unknown="error", categories=Y)
    with pytest.raises(KeyError):
        enc.fit(X)

    enc = OneHotEncoder(handle_unknown="ignore", categories=Y)
    enc.fit(X)


@pytest.mark.mg
def test_onehot_transform_handle_unknown(client):
    X = DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    Y = DataFrame({"chars": ["c", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)
    Y = dask_cudf.from_cudf(Y, npartitions=2)

    enc = OneHotEncoder(handle_unknown="error", sparse_output=False)
    enc = enc.fit(X)
    with pytest.raises(KeyError):
        enc.transform(Y).compute()

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc = enc.fit(X)
    ohe = enc.transform(Y)
    ref = cp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    cp.testing.assert_array_equal(ohe.compute(), ref)


@pytest.mark.mg
def test_onehot_inverse_transform_handle_unknown(client):
    X = DataFrame({"chars": ["a", "b"], "int": [0, 2]})
    X = dask_cudf.from_cudf(X, npartitions=2)
    Y_ohe = cp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    Y_ohe = da.from_array(Y_ohe)

    enc = OneHotEncoder(handle_unknown="ignore")
    enc = enc.fit(X)
    df = enc.inverse_transform(Y_ohe)
    ref = DataFrame({"chars": [None, "b"], "int": [0, 2]})
    assert_frame_equal(df.compute().to_pandas(), ref.to_pandas())


@pytest.mark.mg
@pytest.mark.parametrize("drop", [None, "first"])
@pytest.mark.parametrize("as_array", [True, False], ids=["cupy", "cudf"])
@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
@pytest.mark.parametrize("n_samples", [10, 1000, stress_param(50000)])
def test_onehot_random_inputs(client, drop, as_array, sparse, n_samples):
    X, ary = generate_inputs_from_categories(
        n_samples=n_samples, as_array=as_array
    )
    if as_array:
        dX = da.from_array(X)
    else:
        dX = dask_cudf.from_cudf(X, npartitions=1)

    enc = OneHotEncoder(sparse_output=sparse, drop=drop, categories="auto")
    sk_enc = SkOneHotEncoder(
        sparse_output=sparse, drop=drop, categories="auto"
    )
    ohe = enc.fit_transform(dX)
    ref = sk_enc.fit_transform(ary)
    if sparse:
        cp.testing.assert_array_equal(ohe.compute().toarray(), ref.toarray())
    else:
        cp.testing.assert_array_equal(ohe.compute(), ref)

    inv_ohe = enc.inverse_transform(ohe)
    assert_inverse_equal(inv_ohe.compute(), dX.compute())


@pytest.mark.mg
def test_onehot_drop_idx_first(client):
    X_ary = [["c", 2, "a"], ["b", 2, "b"]]
    X = DataFrame({"chars": ["c", "b"], "int": [2, 2], "letters": ["a", "b"]})
    ddf = dask_cudf.from_cudf(X, npartitions=2)

    enc = OneHotEncoder(sparse_output=False, drop="first")
    sk_enc = SkOneHotEncoder(sparse_output=False, drop="first")
    ohe = enc.fit_transform(ddf)
    ref = sk_enc.fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe.compute(), ref)
    inv = enc.inverse_transform(ohe)
    assert_frame_equal(
        inv.compute().to_pandas().reset_index(drop=True), X.to_pandas()
    )


@pytest.mark.mg
def test_onehot_drop_one_of_each(client):
    X_ary = [["c", 2, "a"], ["b", 2, "b"]]
    X = DataFrame({"chars": ["c", "b"], "int": [2, 2], "letters": ["a", "b"]})
    ddf = dask_cudf.from_cudf(X, npartitions=2)

    drop = dict({"chars": "b", "int": 2, "letters": "b"})
    enc = OneHotEncoder(sparse_output=False, drop=drop)
    sk_enc = SkOneHotEncoder(sparse_output=False, drop=["b", 2, "b"])
    ohe = enc.fit_transform(ddf)
    ref = sk_enc.fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe.compute(), ref)
    inv = enc.inverse_transform(ohe)
    assert_frame_equal(
        inv.compute().to_pandas().reset_index(drop=True), X.to_pandas()
    )


@pytest.mark.mg
@pytest.mark.parametrize(
    "drop, pattern",
    [
        [dict({"chars": "b"}), "`drop` should have as many columns"],
        [
            dict({"chars": "b", "int": [2, 0]}),
            "Trying to drop multiple values",
        ],
        [
            dict({"chars": "b", "int": 3}),
            "Some categories [a-zA-Z, ]* were not found",
        ],
        [
            DataFrame({"chars": "b", "int": 3}),
            "Wrong input for parameter `drop`.",
        ],
    ],
)
def test_onehot_drop_exceptions(client, drop, pattern):
    X = DataFrame({"chars": ["c", "b", "d"], "int": [2, 1, 0]})
    X = dask_cudf.from_cudf(X, npartitions=2)

    with pytest.raises(ValueError, match=pattern):
        OneHotEncoder(sparse_output=False, drop=drop).fit(X)


@pytest.mark.mg
def test_onehot_get_categories(client):
    X = DataFrame({"chars": ["c", "b", "d"], "ints": [2, 1, 0]})
    X = dask_cudf.from_cudf(X, npartitions=2)

    ref = [np.array(["b", "c", "d"]), np.array([0, 1, 2])]
    enc = OneHotEncoder().fit(X)
    cats = enc.categories_

    for i in range(len(ref)):
        np.testing.assert_array_equal(ref[i], cats[i].to_numpy())
