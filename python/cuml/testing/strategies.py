# Copyright (c) 2022, NVIDIA CORPORATION.
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
from hypothesis.strategies import booleans, composite, floats, integers, sampled_from
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@composite
def datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=0, max_value=200),
    n_features=integers(min_value=0, max_value=200),
):
    """
    Generic datasets that can serve as an input to an estimator.

    Parameters
    ----------
    dtypes: SearchStrategy[np.dtype]
        Returned arrays will have a dtype drawn from these types.
    n_samples: SearchStrategy[int]
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int]
        Returned arrays will have number of columns drawn from these values.

    Returns
    -------
    X: SearchStrategy[array] (n_samples, n_features)
        The search strategy for input samples.
    y: SearchStrategy[array] (n_samples,) or (n_samples, n_targets)
        The search strategy for output samples.

    """
    xs = draw(n_samples)
    ys = draw(n_features)
    X = arrays(dtype=dtypes, shape=(xs, ys))
    y = arrays(dtype=dtypes, shape=(xs, draw(sampled_from((1, ys)))))
    return draw(X), draw(y)


@composite
def split_datasets(
    draw,
    datasets=datasets(),
    train_sizes=floats(min_value=0.1, max_value=1.0, exclude_max=True),
):
    """
    Split a generic search strategy for datasets into test and train subsets.

    Note: This function uses the sklearn.model_selection.train_test_split
    function.

    See also:
    datasets(): A search strategy for datasets that can serve as input to this
    strategy.

    Parameters
    ----------
    datasets: SearchStrategy[dataset]
        A search strategy for datasets.
    train_sizes: SearchStrategy[float]
        A search strategy for the train size. Must be provided as float and is
        limited by valid inputs to sklearn's train_test_split() function.

    Returns
    -------
    splitting: list, length=2 * len(arrays)
        List with a drawn train-test split of the drawn dataset.
    """
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
    """
    Generic datasets that can serve as an input to an estimator.

    See also:
    datasets(): Generate generic datasets.
    split_datasets(): Split dataset into test-train subsets.

    Parameters
    ----------
    dtypes: SearchStrategy[np.dtype]
        Returned arrays will have a dtype drawn from these types.
    n_samples: SearchStrategy[int]
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int]
        Returned arrays will have number of columns drawn from these values.
    n_informatives: SearchStrategy[int]
        Determines the number of informative features in a normal dataset.
    is_normal: SearchStrategy[bool]
        Whether the returned dataset is considered normal or more random.

    Returns
    -------
    (X, y):  tuple(SearchStrategy[array], SearchStrategy[array])
        A tuple of search strategies for the requested arrays.
    """
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
