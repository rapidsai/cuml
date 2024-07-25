# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from cuml.internals.array import CumlArray
from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
from hypothesis import assume
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    floating_dtypes,
    integer_dtypes,
)
from hypothesis.strategies import (
    composite,
    integers,
    just,
    none,
    one_of,
    sampled_from,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


_CUML_ARRAY_INPUT_TYPES = ["numpy", "cupy", "series"]


_CUML_ARRAY_DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

_CUML_ARRAY_ORDERS = ["F", "C"]


_CUML_ARRAY_OUTPUT_TYPES = [
    "cudf",
    "cupy",
    "dataframe",
    "numba",
    "numpy",
    "series",
]

# TODO(wphicks): Once all memory types are supported, just directly
# iterate on the enum values
_CUML_ARRAY_MEM_TYPES = ("host", "device")


UNSUPPORTED_CUDF_DTYPES = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
]


@composite
def cuml_array_input_types(draw):
    """Generates all supported cuml array input types."""
    return draw(sampled_from(_CUML_ARRAY_INPUT_TYPES))


@composite
def cuml_array_output_types(draw):
    """Generates all cuml array supported output types."""
    return draw(sampled_from(_CUML_ARRAY_OUTPUT_TYPES))


@composite
def cuml_array_output_dtypes(draw):
    """Generates all cuml array supported output dtypes."""
    return draw(sampled_from(_CUML_ARRAY_DTYPES))


@composite
def cuml_array_dtypes(draw):
    """Generates all supported cuml array dtypes."""
    return draw(sampled_from(_CUML_ARRAY_DTYPES))


@composite
def cuml_array_orders(draw):
    """Generates all supported cuml array orders."""
    return draw(sampled_from(_CUML_ARRAY_ORDERS))


@composite
def cuml_array_mem_types(draw):
    """Generates all supported cuml array mem_types.

    Note that we do not currently test managed memory because it is not yet
    officially supported.
    """
    return draw(sampled_from(_CUML_ARRAY_MEM_TYPES))


@composite
def cuml_array_shapes(
    draw, *, min_dims=1, max_dims=2, min_side=1, max_side=None
):
    """
    Generates cuml array shapes.

    See also: hypothesis.extra.numpy.array_shapes()

    Parameters
    ----------
    min_dims: int, default=1
        Returned shapes will have at least this number of dimensions.
    max_dims: int, default=2
        Returned shapes will have at most this number of dimensions.
    min_side: int, default=1
        Returned shapes will have at least this size in any dimension.
    max_side: int | None, default=min_side + 9
        Returned shapes will have at most this size in any dimension.

    Returns
    -------
    Shapes for cuml array inputs.
    """
    max_side = min_side + 9 if max_side is None else max_side

    if not (1 <= min_dims <= max_dims):
        raise ValueError(
            "Arguments violate condition 1 <= min_dims <= max_dims."
        )
    if not (0 < min_side < max_side):
        raise ValueError(
            "Arguments violate condition 0 < min_side < max_side."
        )

    shapes = array_shapes(
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )
    just_size = integers(min_side, max_side)
    return draw(one_of(shapes, just_size))


def create_cuml_array_input(input_type, dtype, shape, order):
    """
    Creates a valid cuml array input.

    Parameters
    ----------
    input_type: str | None, default=cupy
        Valid input types are "numpy", "cupy", "series".
    dtype: Data type specifier
        A numpy/cupy compatible data type, e.g., numpy.float64.
    shape: int | tuple[int]
        Dimensions of the array to generate.
    order : str in {'C', 'F'}
        Order of arrays to generate, either F- or C-major.

    Returns
    -------
    A cuml array input array.
    """

    input_type = "cupy" if input_type is None else input_type

    multidimensional = isinstance(shape, tuple) and len(shape) > 1
    assume(
        not (
            input_type == "series"
            and (dtype in UNSUPPORTED_CUDF_DTYPES or multidimensional)
        )
    )

    array = cp.ones(shape, dtype=dtype, order=order)

    if input_type == "numpy":
        return np.array(cp.asnumpy(array), dtype=dtype, order=order)

    elif input_type == "series":
        return cudf.Series(array)

    elif input_type == "cupy":
        return array

    raise ValueError(
        "The value for 'input_type' must be "
        f"one of {', '.join(_CUML_ARRAY_INPUT_TYPES)}."
    )


@composite
def cuml_array_inputs(
    draw,
    input_types=cuml_array_input_types(),
    dtypes=cuml_array_dtypes(),
    shapes=cuml_array_shapes(),
    orders=cuml_array_orders(),
):
    """
    Generates valid inputs for cuml arrays.

    Parameters
    ----------
    input_types: SearchStrategy[("numpy", "cupy", "series")], \
        default=cuml_array_input_tyes()
        A search strategy for the type of array input.
    dtypes: SearchStrategy[np.dtype], default=cuml_array_dtypes()
        A search strategy for a numpy/cupy compatible data type.
    shapes: SearchStrategy[int | tuple[int]], default=cuml_array_shapes()
        A search strategy for array shapes.
    orders : str in {'C', 'F'}, default=cuml_array_orders()
        A search strategy for array orders.

    Returns
    -------
    A strategy for valid cuml array inputs.
    """
    input_type = draw(input_types)
    dtype = draw(dtypes)
    shape = draw(shapes)
    order = draw(orders)
    multidimensional = isinstance(shape, tuple) and len(shape) > 1
    assume(
        not (
            input_type == "series"
            and (dtype in UNSUPPORTED_CUDF_DTYPES or multidimensional)
        )
    )

    data = draw(arrays(dtype=dtype, shape=shape))

    if input_type == "numpy":
        ret = np.asarray(data, order=order)

    elif input_type == "cupy":
        ret = cp.array(data, dtype=dtype, order=order)

    elif input_type == "series":
        ret = cudf.Series(data)

    else:
        raise ValueError(
            "The value for 'input_type' must be "
            f"one of {', '.join(_CUML_ARRAY_INPUT_TYPES)}."
        )

    # Cupy currently does not support masked arrays.
    cai = getattr(ret, "__cuda_array_interface__", dict())
    assume(cai.get("mask") is None)

    return ret


@composite
def cuml_arrays(
    draw,
    input_types=cuml_array_input_types(),
    dtypes=cuml_array_dtypes(),
    shapes=cuml_array_shapes(),
    orders=cuml_array_orders(),
    mem_types=cuml_array_mem_types(),
):
    """
    Generates cuml arrays.

    Parameters
    ----------
    input_types: SearchStrategy[("numpy", "cupy", "series")], \
        default=cuml_array_input_tyes()
        A search strategy for the type of array input.
    dtypes: SearchStrategy[np.dtype], default=cuml_array_dtypes()
        A search strategy for a numpy/cupy compatible data type.
    shapes: SearchStrategy[int | tuple[int]], default=cuml_array_shapes()
        A search strategy for array shapes.
    orders : str in {'C', 'F'}, default=cuml_array_orders()
        A search strategy for array orders.

    Returns
    -------
    A strategy for cuml arrays.
    """
    array_input = create_cuml_array_input(
        input_type=draw(input_types),
        dtype=draw(dtypes),
        shape=draw(shapes),
        order=draw(orders),
    )
    return CumlArray(data=array_input, mem_type=draw(mem_types))


def _get_limits(strategy):
    """Try to find the strategy's limits.

    Raises AttributeError if limits cannot be determined.
    """
    # unwrap if lazy
    strategy = getattr(strategy, "wrapped_strategy", strategy)

    try:
        yield getattr(strategy, "value")  # just(...)
    except AttributeError:
        # assume numbers strategy
        yield strategy.start
        yield strategy.stop


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
    n_samples: SearchStrategy[int], \
        default=integers(min_value=0, max_value=200)
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int], \
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


def combined_datasets_strategy(*datasets, name=None, doc=None):
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
        Defaults to a combination of names of the provided dataset strategy
        functions.
    doc: The doc-string of the returned search strategy, default=None
        Defaults to a generic doc-string.

    Returns
    -------
    Datasets search strategy: SearchStrategy[array], SearchStrategy[array]
    """

    @composite
    def strategy(
        draw,
        dtypes=floating_dtypes(),
        n_samples=integers(min_value=1, max_value=200),
        n_features=integers(min_value=1, max_value=200),
    ):
        """Datasets strategy composed of multiple datasets strategies."""
        datasets_strategies = (
            dataset(dtypes, n_samples, n_features) for dataset in datasets
        )
        return draw(one_of(datasets_strategies))

    strategy.__name__ = "datasets" if name is None else name
    if doc is not None:
        strategy.__doc__ = doc

    return strategy


@composite
def split_datasets(
    draw,
    datasets,
    test_sizes=None,
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
    test_sizes: SearchStrategy[int | float], default=None
        A search strategy for the test size. Must be provided as a search
        strategy for integers or floats. Integers should be bound by one and
        the sample size, floats should be between 0 and 1.0. Defaults to
        a search strategy that will generate a valid unbiased split.

    Returns
    -------
    (X_train, X_test, y_train, y_test): tuple[SearchStrategy[array], ...]
        The train-test split of the input and output samples drawn from
        the provided datasets search strategy.
    """
    X, y = draw(datasets)
    assume(len(X) > 1)

    # Determine default value for test_sizes
    if test_sizes is None:
        test_sizes = integers(1, max(1, len(X) - 1))

    test_size = draw(test_sizes)

    # Check assumptions for test_size
    if isinstance(test_size, float):
        assume(int(len(X) * test_size) > 0)
        assume(int(len(X) * (1.0 - test_size)) > 0)
    elif isinstance(test_size, int):
        assume(1 < test_size < len(X))

    return train_test_split(X, y, test_size=test_size)


@composite
def standard_regression_datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=100, max_value=200),
    n_features=integers(min_value=100, max_value=200),
    *,
    n_informative=None,
    n_targets=just(1),
    bias=just(0.0),
    effective_rank=none(),
    tail_strength=just(0.5),
    noise=just(0.0),
    shuffle=just(True),
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
    (X, y):  SearchStrategy[array], SearchStrategy[array]
        A tuple of search strategies for arrays subject to the constraints of
        the provided parameters.
    """
    n_features_ = draw(n_features)
    if n_informative is None:
        n_informative = just(max(min(n_features_, 1), int(0.1 * n_features_)))
    X, y = make_regression(
        n_samples=draw(n_samples),
        n_features=n_features_,
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
    standard_datasets,
    standard_regression_datasets,
    name="regression_datasets",
    doc="""
    Returns strategy for the generation of regression problem datasets.

    Drawn from the standard_datasets and the standard_regression_datasets
    strategies.
    """,
)


@composite
def standard_classification_datasets(
    draw,
    dtypes=floating_dtypes(),
    n_samples=integers(min_value=100, max_value=200),
    n_features=integers(min_value=10, max_value=20),
    *,
    n_informative=None,
    n_redundant=None,
    n_repeated=just(0),
    n_classes=just(2),
    n_clusters_per_class=just(2),
    weights=none(),
    flip_y=just(0.01),
    class_sep=just(1.0),
    hypercube=just(True),
    shift=just(0.0),
    scale=just(1.0),
    shuffle=just(True),
    random_state=None,
    labels_dtypes=integer_dtypes(),
):
    n_features_ = draw(n_features)
    if n_informative is None:
        try:
            # Try to meet:
            #   log_2(n_classes * n_clusters_per_class) <= n_informative
            n_classes_min = min(_get_limits(n_classes))
            n_clusters_per_class_min = min(_get_limits(n_clusters_per_class))
            n_informative_min = int(
                np.ceil(np.log2(n_classes_min * n_clusters_per_class_min))
            )
        except AttributeError:
            # Otherwise aim for 10% of n_features, but at least 1.
            n_informative_min = max(1, int(0.1 * n_features_))

        n_informative = just(min(n_features_, n_informative_min))
    if n_redundant is None:
        n_redundant = just(max(min(n_features_, 1), int(0.1 * n_features_)))

    # Check whether the
    #   log_2(n_classes * n_clusters_per_class) <= n_informative
    # inequality can in principle be met.
    try:
        n_classes_min = min(_get_limits(n_classes))
        n_clusters_per_class_min = min(_get_limits(n_clusters_per_class))
        n_informative_max = max(_get_limits(n_informative))
    except AttributeError:
        pass  # unable to determine limits
    else:
        if (
            np.log2(n_classes_min * n_clusters_per_class_min)
            > n_informative_max
        ):
            raise ValueError(
                "Assumptions cannot be met, the following inequality must "
                "hold: log_2(n_classes * n_clusters_per_class) "
                "<= n_informative ."
            )

    # Check base assumption concerning the composition of feature vectors.
    n_informative_ = draw(n_informative)
    n_redundant_ = draw(n_redundant)
    n_repeated_ = draw(n_repeated)
    assume(n_informative_ + n_redundant_ + n_repeated_ <= n_features_)

    # Check base assumption concerning relationship of number of clusters and
    # informative features.
    n_classes_ = draw(n_classes)
    n_clusters_per_class_ = draw(n_clusters_per_class)
    assume(np.log2(n_classes_ * n_clusters_per_class_) <= n_informative_)

    X, y = make_classification(
        n_samples=draw(n_samples),
        n_features=n_features_,
        n_informative=n_informative_,
        n_redundant=n_redundant_,
        n_repeated=n_repeated_,
        n_classes=n_classes_,
        n_clusters_per_class=n_clusters_per_class_,
        weights=draw(weights),
        flip_y=draw(flip_y),
        class_sep=draw(class_sep),
        hypercube=draw(hypercube),
        shift=draw(shift),
        scale=draw(scale),
        shuffle=draw(shuffle),
        random_state=random_state,
    )
    return X.astype(draw(dtypes)), y.astype(draw(labels_dtypes))
