# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from hypothesis import assume
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis.strategies import booleans, composite, floats, integers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@composite
def datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=0, max_value=200),
    n_features=integers(min_value=0, max_value=200),
):
    xs = draw(n_samples)
    ys = draw(n_features)
    X = arrays(dtype=dtypes, shape=(xs, ys))
    y = arrays(dtype=dtypes, shape=(xs, 1))
    return draw(X), draw(y)


@composite
def split_datasets(
    draw,
    datasets=datasets(),
    train_sizes=floats(min_value=0.1, max_value=1.0, exclude_max=True),
):
    X, y = draw(datasets)
    ts = draw(train_sizes)
    assume(int(len(X) * ts) > 0)  # train_test_split limitation
    return train_test_split(X, y, train_size=ts)


@composite
def regression_datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=0, max_value=200),
    n_features=integers(min_value=0, max_value=200),
    n_informatives=integers(min_value=0, max_value=200),
    is_normal=booleans(),
):
    if draw(is_normal):
        dtype_ = draw(dtypes)
        X, y = make_regression(
            n_samples=draw(n_samples),
            n_features=draw(n_features),
            n_informative=draw(n_informatives),
        )
        return X.astype(dtype_), y.astype(dtype_)
    else:
        return draw(
            datasets(
                dtypes=dtypes,
                n_samples=n_samples,
                n_features=n_features,
            )
        )
