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

import pytest
import cudf
from cuml.preprocessing.model_selection import train_test_split


@pytest.mark.parametrize("n_rows", [100, 100000])
@pytest.mark.parametrize("train_size", [0.0, 0.1, 0.5, 0.75, 1.0])
def test_split(n_rows, train_size):
    X = cudf.DataFrame({"x": range(n_rows)})
    y = cudf.Series(([0] * (n_rows // 2)) + ([1] * (n_rows // 2)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size
    )
    assert len(X_train) == len(y_train) == pytest.approx(train_size * len(X))
    assert (
        len(X_test) == len(y_test) == pytest.approx((1 - train_size) * len(X))
    )

    X_reconstructed = cudf.multi.concat([X_train, X_test]).sort_values(
        by=["x"]
    )
    y_reconstructed = y_train.append(y_test).sort_values()

    assert all(X_reconstructed == X)
    assert all(y_reconstructed == y)


@pytest.mark.parametrize("n_rows", [100, 100000])
def test_split_column(n_rows):
    data = cudf.DataFrame(
        {
            "x": range(n_rows),
            "y": ([0] * (n_rows // 2)) + ([1] * (n_rows // 2)),
        }
    )
    train_size = 0.8

    X_train, X_test, y_train, y_test = train_test_split(
        data, "y", train_size=train_size
    )

    assert (
        len(X_train) == len(y_train) == pytest.approx(train_size * len(data))
    )
    assert (
        len(X_test)
        == len(y_test)
        == pytest.approx((1 - train_size) * len(data))
    )

    X_reconstructed = cudf.multi.concat([X_train, X_test]).sort_values(
        by=["x"]
    )
    y_reconstructed = y_train.append(y_test).sort_values()

    assert all(data == X_reconstructed.assign(y=y_reconstructed))


def test_split_size_mismatch():
    X = cudf.DataFrame({'x': range(3)})
    y = cudf.Series([0, 1])

    with pytest.raises(ValueError):
        train_test_split(X, y)


@pytest.mark.parametrize('train_size', [1.2, 100])
def test_split_invalid_proportion(train_size):
    X = cudf.DataFrame({'x': range(10)})
    y = cudf.Series([0] * 10)

    with pytest.raises(ValueError):
        train_test_split(X, y, train_size=train_size)
