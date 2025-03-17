# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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


from cuml.testing.test_preproc_utils import assert_allclose
from sklearn.preprocessing import (
    StandardScaler as skStandardScaler,
    Normalizer as skNormalizer,
    PolynomialFeatures as skPolynomialFeatures,
    OneHotEncoder as skOneHotEncoder,
)
from cuml.preprocessing import (
    StandardScaler as cuStandardScaler,
    Normalizer as cuNormalizer,
    PolynomialFeatures as cuPolynomialFeatures,
    OneHotEncoder as cuOneHotEncoder,
)
from sklearn.compose import (
    ColumnTransformer as skColumnTransformer,
    make_column_transformer as sk_make_column_transformer,
    make_column_selector as sk_make_column_selector,
)
from cuml.compose import (
    ColumnTransformer as cuColumnTransformer,
    make_column_transformer as cu_make_column_transformer,
    make_column_selector as cu_make_column_selector,
)
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.internals.safe_imports import cpu_only_import_from
from cuml.internals.safe_imports import cpu_only_import
import pytest

from cuml.internals.safe_imports import gpu_only_import

from cuml.testing.test_preproc_utils import (  # noqa: F401
    clf_dataset,
    sparse_clf_dataset,
)

cudf = gpu_only_import("cudf")
np = cpu_only_import("numpy")
pdDataFrame = cpu_only_import_from("pandas", "DataFrame")
cuDataFrame = gpu_only_import_from("cudf", "DataFrame")


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize(
    "transformer_weights", [None, {"scaler": 2.4, "normalizer": 1.8}]
)
def test_column_transformer(
    clf_dataset, remainder, transformer_weights  # noqa: F811
):
    X_np, X = clf_dataset

    sk_selec1 = [0, 2]
    sk_selec2 = [1, 3]
    cu_selec1 = sk_selec1
    cu_selec2 = sk_selec2
    if isinstance(X, (pdDataFrame, cuDataFrame)):
        cu_selec1 = ["c" + str(i) for i in sk_selec1]
        cu_selec2 = ["c" + str(i) for i in sk_selec2]

    cu_transformers = [
        ("scaler", cuStandardScaler(), cu_selec1),
        ("normalizer", cuNormalizer(), cu_selec2),
    ]

    transformer = cuColumnTransformer(
        cu_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
    )
    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    assert type(t_X) is type(X)

    sk_transformers = [
        ("scaler", skStandardScaler(), sk_selec1),
        ("normalizer", skNormalizer(), sk_selec2),
    ]

    transformer = skColumnTransformer(
        sk_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize(
    "transformer_weights", [None, {"scaler": 2.4, "normalizer": 1.8}]
)
@pytest.mark.parametrize("sparse_threshold", [0.2, 0.8])
def test_column_transformer_sparse(
    sparse_clf_dataset,
    remainder,  # noqa: F811
    transformer_weights,
    sparse_threshold,
):
    X_np, X = sparse_clf_dataset

    if X.format == "csc":
        pytest.xfail()
    dataset_density = X.nnz / X.size

    cu_transformers = [
        ("scaler", cuStandardScaler(with_mean=False), [0, 2]),
        ("normalizer", cuNormalizer(), [1, 3]),
    ]

    transformer = cuColumnTransformer(
        cu_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
        sparse_threshold=sparse_threshold,
    )
    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    if dataset_density < sparse_threshold:
        # Sparse input -> sparse output if dataset_density > sparse_threshold
        # else sparse input -> dense output
        assert type(t_X) is type(X)

    sk_transformers = [
        ("scaler", skStandardScaler(with_mean=False), [0, 2]),
        ("normalizer", skNormalizer(), [1, 3]),
    ]

    transformer = skColumnTransformer(
        sk_transformers,
        remainder=remainder,
        transformer_weights=transformer_weights,
        sparse_threshold=sparse_threshold,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_make_column_transformer(clf_dataset, remainder):  # noqa: F811
    X_np, X = clf_dataset

    sk_selec1 = [0, 2]
    sk_selec2 = [1, 3]
    cu_selec1 = sk_selec1
    cu_selec2 = sk_selec2
    if isinstance(X, (pdDataFrame, cuDataFrame)):
        cu_selec1 = ["c" + str(i) for i in sk_selec1]
        cu_selec2 = ["c" + str(i) for i in sk_selec2]

    transformer = cu_make_column_transformer(
        (cuStandardScaler(), cu_selec1),
        (cuNormalizer(), cu_selec2),
        remainder=remainder,
    )

    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    assert type(t_X) is type(X)

    transformer = sk_make_column_transformer(
        (skStandardScaler(), sk_selec1),
        (skNormalizer(), sk_selec2),
        remainder=remainder,
    )
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize("sparse_threshold", [0.2, 0.8])
def test_make_column_transformer_sparse(
    sparse_clf_dataset, remainder, sparse_threshold  # noqa: F811
):
    X_np, X = sparse_clf_dataset

    if X.format == "csc":
        pytest.xfail()
    dataset_density = X.nnz / X.size

    transformer = cu_make_column_transformer(
        (cuStandardScaler(with_mean=False), [0, 2]),
        (cuNormalizer(), [1, 3]),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
    )

    ft_X = transformer.fit_transform(X)
    t_X = transformer.transform(X)
    if dataset_density < sparse_threshold:
        # Sparse input -> sparse output if dataset_density > sparse_threshold
        # else sparse input -> dense output
        assert type(t_X) is type(X)

    transformer = sk_make_column_transformer(
        (skStandardScaler(with_mean=False), [0, 2]),
        (skNormalizer(), [1, 3]),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
    )

    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(ft_X, sk_t_X)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.skip(
    reason="scikit-learn replaced get_feature_names with "
    "get_feature_names_out"
    "https://github.com/rapidsai/cuml/issues/5159"
)
def test_column_transformer_get_feature_names(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    cu_transformers = [("PolynomialFeatures", cuPolynomialFeatures(), [0, 2])]
    transformer = cuColumnTransformer(cu_transformers)
    transformer.fit_transform(X)
    cu_feature_names = transformer.get_feature_names()

    sk_transformers = [("PolynomialFeatures", skPolynomialFeatures(), [0, 2])]
    transformer = skColumnTransformer(sk_transformers)
    transformer.fit_transform(X_np)
    sk_feature_names = transformer.get_feature_names()

    assert cu_feature_names == sk_feature_names


def test_column_transformer_named_transformers_(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    cu_transformers = [("PolynomialFeatures", cuPolynomialFeatures(), [0, 2])]
    transformer = cuColumnTransformer(cu_transformers)
    transformer.fit_transform(X)
    cu_named_transformers = transformer.named_transformers_

    sk_transformers = [("PolynomialFeatures", skPolynomialFeatures(), [0, 2])]
    transformer = skColumnTransformer(sk_transformers)
    transformer.fit_transform(X_np)
    sk_named_transformers = transformer.named_transformers_

    assert cu_named_transformers.keys() == sk_named_transformers.keys()


def test_make_column_selector():
    X_np = pdDataFrame(
        {
            "city": ["London", "London", "Paris", "Sallisaw"],
            "rating": [5, 3, 4, 5],
            "temperature": [21.0, 21.0, 24.0, 28.0],
        }
    )
    X = cudf.from_pandas(X_np)

    cu_transformers = [
        (
            "ohe",
            cuOneHotEncoder(),
            cu_make_column_selector(dtype_exclude=np.number),
        ),
        (
            "scaler",
            cuStandardScaler(),
            cu_make_column_selector(dtype_include=np.integer),
        ),
        (
            "normalizer",
            cuNormalizer(),
            cu_make_column_selector(pattern="temp"),
        ),
    ]
    transformer = cuColumnTransformer(cu_transformers, remainder="drop")
    t_X = transformer.fit_transform(X)

    sk_transformers = [
        (
            "ohe",
            skOneHotEncoder(),
            sk_make_column_selector(dtype_exclude=np.number),
        ),
        (
            "scaler",
            skStandardScaler(),
            sk_make_column_selector(dtype_include=np.integer),
        ),
        (
            "normalizer",
            skNormalizer(),
            sk_make_column_selector(pattern="temp"),
        ),
    ]
    transformer = skColumnTransformer(sk_transformers, remainder="drop")
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)
    assert type(t_X) is type(X)


def test_column_transformer_index(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    if not isinstance(X, (pdDataFrame, cuDataFrame)):
        pytest.skip()

    cu_transformers = [("scaler", cuStandardScaler(), X.columns)]

    transformer = cuColumnTransformer(cu_transformers)
    transformer.fit_transform(X)
