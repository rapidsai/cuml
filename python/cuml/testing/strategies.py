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
from hypothesis.strategies import (composite, floats, integers, just, none,
                                   one_of)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@composite
def standard_datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=0, max_value=200),
    n_features=integers(min_value=0, max_value=200),
    *,
    n_targets=just(1),
):
    """
    Returns a strategy to generate standard estimator input datasets.

    Parameters
    ----------
    dtypes: SearchStrategy[np.dtype], default=floating_dtypes()
        Returned arrays will have a dtype drawn from these types.
    n_samples: SearchStrategy[int],\
        default=integers(min_value=0, max_value=200)
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int],\
        default=integers(min_value=0, max_values=200)
        Returned arrays will have number of columns drawn from these values.
    n_targets: SearchStrategy[int], default=just(1)
        Determines the number of targets returned datasets may contain.

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
    y = arrays(dtype=dtypes, shape=(xs, draw(n_targets)))
    return draw(X), draw(y)


def combined_datasets_strategy(* datasets, name=None, doc=None):
    """
    Combine multiple datasets strategies into a single datasets strategy.

    This function will return a new strategy that will build the provided
    strategy functions with the common parameters (dtypes, n_samples,
    n_features) and then draw from one of them.

    Parameters:
    -----------
    * datasets: list[Callable[[dtypes, n_samples, n_features], SearchStrategy]]
        A list of functions that return a dataset search strategy when called
        with the shown arguments.
    name: The name of the returned search strategy, default="datasets"
        Defaults to a combination of names of the provided dataset stratgegy
        functions.
    doc: The doc-string of the returned search strategy, default=None
        Defaults to a generic doc-string.

    Returns
    -------
    Datasets search strategy: SearchStrategy[array, array]
    """

    @composite
    def strategy(
        draw,
        dtypes=floating_dtypes(),
        n_samples=integers(min_value=0, max_value=200),
        n_features=integers(min_value=0, max_value=200)
    ):
        """Datasets strategy composed of multiple datasets strategies."""
        datasets_strategies = (
            dataset(dtypes, n_samples, n_features) for dataset in datasets)
        return draw(one_of(datasets_strategies))

    strategy.__name__ = "datasets" if name is None else name
    if doc is not None:
        strategy.__doc__ = doc

    return strategy


@composite
def split_datasets(
    draw,
    datasets,
    test_sizes=floats(min_value=0.25, max_value=1.0, exclude_max=True),
):
    """
    Split a generic search strategy for datasets into test and train subsets.

    The resulting split is guaranteed to have at least one sample in both the
    train and test split respectively.

    Note: This function uses the sklearn.model_selection.train_test_split
    function.

    See also:
    standard_datasets(): A search strategy for datasets that can serve as input
    to this strategy.

    Parameters
    ----------
    datasets: SearchStrategy[dataset]
        A search strategy for datasets.
    test_sizes: SearchStrategy[float],
        default=floats(min_value=0.25, max_value=0.1, exclude_max=True)
        A search strategy for the test size. Must be provided as float and is
        limited by valid inputs to sklearn's train_test_split() function.

    Returns
    -------
    (X_train, X_test, y_train, y_test): SearchStrategy[4 * array]
        The train-test split of the input and output samples drawn from
        the provided datasets search strategy.
    """
    X, y = draw(datasets)
    test_size = draw(test_sizes)
    assume(int(len(X) * test_size) > 0)
    assume(int(len(X) * (1.0 - test_size)) > 0)
    return train_test_split(X, y, test_size=test_size)


@composite
def standard_regression_datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=0, max_value=100),
    n_features=integers(min_value=0, max_value=100),
    *,
    n_informative=None,
    n_targets=just(1),
    bias=just(0.0),
    effective_rank=none(),
    tail_strength=just(0.5),
    noise=just(0.0),
    shuffle=just(True),
    coef=just(False),
    random_state=None,
):
    """
    Returns a strategy to generate regression problem input datasets.

    Note:
    This function uses the sklearn.datasets.make_regression function to
    generate the regression problem from the provided search strategies.


    Parameters
    ----------
    dtypes: SearchStrategy[np.dtype]
        Returned arrays will have a dtype drawn from these types.
    n_samples: SearchStrategy[int]
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int]
        Returned arrays will have number of columns drawn from these values.
    n_informative: SearchStrategy[int], default=none
        A search strategy for the number of informative features. If none,
        will use 10% of the actual number of features, but not less than 1
        unless the number of features is zero.
    n_targets: SearchStrategy[int], default=just(1)
        A search strategy for the number of targets, that means the number of
        columns of the returned y output array.
    bias: SearchStrategy[float], default=just(0.0)
        A search strategy for the bias term.
    effective_rank=none()
        If not None, a search strategy for the effective rank of the input data
        for the regression problem. See sklearn.dataset.make_regression() for a
        detailed explanation of this parameter.
    tail_strength: SearchStrategy[float], default=just(0.5)
        See sklearn.dataset.make_regression() for a detailed explanation of
        this parameter.
    noise: SearchStrategy[float], default=just(0.0)
        A search strategy for the standard deviation of the gaussian noise.
    shuffle: SearchStrategy[bool], default=just(True)
        A boolean search strategy to determine whether samples and features
        are shuffled.
    random_state: int, RandomState instance or None, default=None
        Pass a random state or integer to determine the random number
        generation for data set generation.

    Returns
    -------
    (X, y):  SearchStrategy[array, array]
        A search strategy for a tuple of two arrays subject to the
        provided parameters.
    """
    n_samples_ = draw(n_samples)
    if n_informative is None:
        n_informative = just(max(min(n_samples_, 1), int(0.1 * n_samples_)))
    X, y = make_regression(
        n_samples=n_samples_,
        n_features=draw(n_features),
        n_informative=draw(n_informative),
        n_targets=draw(n_targets),
        bias=draw(bias),
        effective_rank=draw(effective_rank),
        tail_strength=draw(tail_strength),
        noise=draw(noise),
        shuffle=draw(shuffle),
        random_state=random_state,
    )
    dtype_ = draw(dtypes)
    return X.astype(dtype_), y.astype(dtype_)


regression_datasets = combined_datasets_strategy(
    standard_datasets, standard_regression_datasets,
    name="regression_datasets",
    doc="""
    Returns strategy for the generation of regression problem datasets.

    Drawn from the standard_datasets and the standard_regression_datasets
    strategies.
    """
)
