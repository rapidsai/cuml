#
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
import dask.array as da
from cuml.dask.metrics import confusion_matrix
from cuml.testing.utils import stress_param, generate_random_labels
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pytest
from cuml.internals.safe_imports import gpu_only_import
from itertools import chain, permutations

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


@pytest.mark.mg
@pytest.mark.parametrize("chunks", ["auto", 2, 1])
def test_confusion_matrix(client, chunks):
    y_true = da.from_array(cp.array([2, 0, 2, 2, 0, 1]), chunks=chunks)
    y_pred = da.from_array(cp.array([0, 0, 2, 2, 0, 2]), chunks=chunks)
    cm = confusion_matrix(y_true, y_pred)
    ref = cp.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    cp.testing.assert_array_equal(cm, ref)


@pytest.mark.mg
@pytest.mark.parametrize("chunks", ["auto", 2, 1])
def test_confusion_matrix_binary(client, chunks):
    y_true = da.from_array(cp.array([0, 1, 0, 1]), chunks=chunks)
    y_pred = da.from_array(cp.array([1, 1, 1, 0]), chunks=chunks)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ref = cp.array([0, 2, 1, 1])
    cp.testing.assert_array_equal(ref, cp.array([tn, fp, fn, tp]))


@pytest.mark.mg
@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_confusion_matrix_random(n_samples, dtype, problem_type, client):
    upper_range = 2 if problem_type == "binary" else 1000

    y_true, y_pred, np_y_true, np_y_pred = generate_random_labels(
        lambda rng: rng.randint(0, upper_range, n_samples).astype(dtype),
        as_cupy=True,
    )
    y_true, y_pred = da.from_array(y_true), da.from_array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    ref = sk_confusion_matrix(np_y_true, np_y_pred)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.mg
@pytest.mark.parametrize(
    "normalize, expected_results",
    [
        ("true", 0.333333333),
        ("pred", 0.333333333),
        ("all", 0.1111111111),
        (None, 2),
    ],
)
def test_confusion_matrix_normalize(normalize, expected_results, client):
    y_test = da.from_array(cp.array([0, 1, 2] * 6))
    y_pred = da.from_array(cp.array(list(chain(*permutations([0, 1, 2])))))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    cp.testing.assert_allclose(cm, cp.array(expected_results))


@pytest.mark.mg
@pytest.mark.parametrize("labels", [(0, 1), (2, 1), (2, 1, 4, 7), (2, 20)])
def test_confusion_matrix_multiclass_subset_labels(labels, client):
    y_true, y_pred, np_y_true, np_y_pred = generate_random_labels(
        lambda rng: rng.randint(0, 3, 10).astype(np.int32), as_cupy=True
    )
    y_true, y_pred = da.from_array(y_true), da.from_array(y_pred)

    ref = sk_confusion_matrix(np_y_true, np_y_pred, labels=labels)
    labels = cp.array(labels, dtype=np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.mg
@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weights_dtype", ["int", "float"])
def test_confusion_matrix_random_weights(
    n_samples, dtype, weights_dtype, client
):
    y_true, y_pred, np_y_true, np_y_pred = generate_random_labels(
        lambda rng: rng.randint(0, 10, n_samples).astype(dtype), as_cupy=True
    )
    y_true, y_pred = da.from_array(y_true), da.from_array(y_pred)

    if weights_dtype == "int":
        sample_weight = np.random.RandomState(0).randint(0, 10, n_samples)
    else:
        sample_weight = np.random.RandomState(0).rand(n_samples)

    ref = sk_confusion_matrix(
        np_y_true, np_y_pred, sample_weight=sample_weight
    )

    sample_weight = cp.array(sample_weight)
    sample_weight = da.from_array(sample_weight)

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)
