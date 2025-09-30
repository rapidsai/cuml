# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
"""Data generators for cuML benchmarks

The main entry point for consumers is gen_data, which
wraps the underlying data generators.

Notes when writing new generators:

Each generator is a function that accepts:
 * n_samples (set to 0 for 'default')
 * n_features (set to 0 for 'default')
 * random_state
 * (and optional generator-specific parameters)

The function should return a 2-tuple (X, y), where X is a Pandas
dataframe and y is a Pandas series. If the generator does not produce
labels, it can return (X, None)

A set of helper functions (convert_*) can convert these to alternative
formats. Future revisions may support generating cudf dataframes or
GPU arrays directly instead.

"""

import functools
import os
from urllib.request import urlretrieve

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.model_selection
from numba import cuda
from sklearn.datasets import fetch_covtype, load_svmlight_file

import cuml.datasets
from cuml.internals import input_utils


def _gen_data_regression(
    n_samples, n_features, random_state=42, dtype=np.float32
):
    """Wrapper for sklearn make_regression"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_features = 100

    X_arr, y_arr = cuml.datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        dtype=dtype,
    )

    X_df = cudf.DataFrame(X_arr)
    y_df = cudf.Series(np.squeeze(y_arr))

    return X_df, y_df


def _gen_data_blobs(
    n_samples, n_features, random_state=42, centers=None, dtype=np.float32
):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100

    X_arr, y_arr = cuml.datasets.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state,
        dtype=dtype,
    )

    return X_arr, y_arr


def _gen_data_zeros(n_samples, n_features, dtype=np.float32):
    """Dummy generator for use in testing - returns all 0s"""
    return cp.zeros((n_samples, n_features), dtype=dtype), cp.zeros(
        n_samples, dtype=dtype
    )


def _gen_data_classification(
    n_samples, n_features, random_state=42, n_classes=2, dtype=np.float32
):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100

    X_arr, y_arr = cuml.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        dtype=dtype,
    )
    X_df = cudf.DataFrame(X_arr)
    y_df = cudf.Series(y_arr)
    return X_df, y_df


# Default location to cache datasets
DATASETS_DIRECTORY = "."


def _gen_data_airline_regression(datasets_root_dir):

    url = "http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"

    local_url = os.path.join(datasets_root_dir, os.path.basename(url))

    cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    dtype = np.float64
    dtype_columns = {
        "Year": dtype,
        "Month": dtype,
        "DayofMonth": dtype,
        "DayofWeek": dtype,
        "CRSDepTime": dtype,
        "CRSArrTime": dtype,
        "FlightNum": dtype,
        "ActualElapsedTime": dtype,
        "Distance": dtype,
        "Diverted": dtype,
        "ArrDelay": dtype,
    }

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    df = pd.read_csv(local_url, names=cols, dtype=dtype_columns)

    # Encode categoricals as numeric
    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    X = df[df.columns.difference(["ArrDelay"])]
    y = df["ArrDelay"]

    return X, y


def _gen_data_airline_classification(datasets_root_dir):
    X, y = _gen_data_airline_regression(datasets_root_dir)
    y = 1 * (y > 0)
    return X, y


def _gen_data_bosch(datasets_root_dir):

    local_url = os.path.join(datasets_root_dir, "train_numeric.csv.zip")

    if not os.path.isfile(local_url):
        raise ValueError(
            "Bosch dataset not found (search path: %s)" % local_url
        )

    df = pd.read_csv(
        local_url, index_col=0, compression="zip", dtype=np.float32
    )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def _gen_data_covtype(datasets_root_dir):

    X, y = fetch_covtype(return_X_y=True)
    # Labele range in covtype start from 1, making it start from 0
    y = y - 1

    X = pd.DataFrame(X)
    y = pd.Series(y)

    return X, y


def _gen_data_epsilon(datasets_root_dir):

    url_train = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
        "/epsilon_normalized.bz2"
    )
    url_test = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
        "/epsilon_normalized.t.bz2"
    )

    local_url_train = os.path.join(
        datasets_root_dir, os.path.basename(url_train)
    )
    local_url_test = os.path.join(
        datasets_root_dir, os.path.basename(url_test)
    )

    if not os.path.isfile(local_url_train):
        urlretrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        urlretrieve(url_test, local_url_test)

    X_train, y_train = load_svmlight_file(local_url_train, dtype=np.float32)
    X_test, y_test = load_svmlight_file(local_url_test, dtype=np.float32)

    X_train = pd.DataFrame(X_train.toarray())
    X_test = pd.DataFrame(X_test.toarray())

    y_train[y_train <= 0] = 0
    y_test[y_test <= 0] = 0
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    return X, y


def _gen_data_fraud(datasets_root_dir):

    local_url = os.path.join(datasets_root_dir, "creditcard.csv.zip")

    if not os.path.isfile(local_url):
        raise ValueError(
            "Fraud dataset not found (search path: %s)" % local_url
        )

    df = pd.read_csv(local_url, dtype=np.float32)
    X = df[[col for col in df.columns if col.startswith("V")]]
    y = df["Class"]

    return X, y


def _gen_data_higgs(datasets_root_dir):

    higgs_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"  # noqa

    local_url = os.path.join(datasets_root_dir, os.path.basename(higgs_url))

    if not os.path.isfile(local_url):
        urlretrieve(higgs_url, local_url)

    col_names = ["label"] + [
        "col-{}".format(i) for i in range(2, 30)
    ]  # Assign column names
    dtypes_ls = [np.int32] + [
        np.float32 for _ in range(2, 30)
    ]  # Assign dtypes to each column

    df = pd.read_csv(
        local_url,
        names=col_names,
        dtype={k: v for k, v in zip(col_names, dtypes_ls)},
    )

    X = df[df.columns.difference(["label"])]
    y = df["label"]

    return X, y


def _gen_data_year(datasets_root_dir):

    year_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"

    local_url = os.path.join(datasets_root_dir, "YearPredictionMSD.txt.zip")

    if not os.path.isfile(local_url):
        urlretrieve(year_url, local_url)

    df = pd.read_csv(local_url, header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    return X, y


def _convert_to_numpy(data):
    """Returns tuple data with all elements converted to numpy ndarrays"""
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_numpy(d) for d in data])
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    elif isinstance(data, cudf.DataFrame):
        return data.to_numpy()
    elif isinstance(data, cudf.Series):
        return data.to_numpy()
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise Exception("Unsupported type %s" % str(type(data)))


def _convert_to_cupy(data):
    """Returns tuple data with all elements converted to cupy ndarrays"""
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_cupy(d) for d in data])
    elif isinstance(data, np.ndarray):
        return cp.asarray(data)
    elif isinstance(data, cp.ndarray):
        return data
    elif isinstance(data, cudf.DataFrame):
        return data.values
    elif isinstance(data, cudf.Series):
        return data.values
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return cp.asarray(data.to_numpy())
    else:
        raise Exception("Unsupported type %s" % str(type(data)))


def _convert_to_cudf(data):
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_cudf(d) for d in data])
    elif isinstance(data, (cudf.DataFrame, cudf.Series)):
        return data
    elif isinstance(data, pd.DataFrame):
        return cudf.DataFrame(data)
    elif isinstance(data, pd.Series):
        return cudf.Series(data)
    elif isinstance(data, np.ndarray):
        data = np.squeeze(data)
        if data.ndim == 1:
            return cudf.Series(data)
        else:
            return cudf.DataFrame(data)
    elif isinstance(data, cp.ndarray):
        data = np.squeeze(cp.asnumpy(data))
        if data.ndim == 1:
            return cudf.Series(data)
        else:
            return cudf.DataFrame(data)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))


def _convert_to_pandas(data):
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_pandas(d) for d in data])
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data
    elif isinstance(data, (cudf.DataFrame, cudf.Series)):
        return data.to_pandas()
    elif isinstance(data, np.ndarray):
        data = np.squeeze(data)
        if data.ndim == 1:
            return pd.Series(data)
        else:
            return pd.DataFrame(data)
    elif isinstance(data, cp.ndarray):
        data = np.squeeze(cp.asnumpy(data))
        if data.ndim == 1:
            return pd.Series(data)
        else:
            return pd.DataFrame(data)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))


def _convert_to_gpuarray(data, order="F"):
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_gpuarray(d, order=order) for d in data])
    elif isinstance(data, pd.DataFrame):
        return _convert_to_gpuarray(cudf.DataFrame(data), order=order)
    elif isinstance(data, pd.Series):
        gs = cudf.Series(data)
        return cuda.as_cuda_array(gs)
    else:
        return input_utils.input_to_cuml_array(data, order=order)[0].to_output(
            "numba"
        )


def _convert_to_gpuarray_c(data):
    return _convert_to_gpuarray(data, order="C")


def _sparsify_and_convert(data, input_type, sparsity_ratio=0.3):
    """Randomly set values to 0 and produce a sparse array."""
    random_loc = np.random.choice(
        data.size, int(data.size * sparsity_ratio), replace=False
    )
    data.ravel()[random_loc] = 0
    if input_type == "csr":
        return scipy.sparse.csr_matrix(data)
    elif input_type == "csc":
        return scipy.sparse.csc_matrix(data)
    else:
        TypeError("Wrong sparse input type {}".format(input_type))


def _convert_to_scipy_sparse(data, input_type):
    """Returns a tuple of arrays. Each of the arrays
    have some of its values being set randomly to 0,
    it is then converted to a scipy sparse array"""
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_scipy_sparse(d, input_type) for d in data])
    elif isinstance(data, np.ndarray):
        return _sparsify_and_convert(data, input_type)
    elif isinstance(data, cudf.DataFrame):
        return _sparsify_and_convert(data.to_numpy(), input_type)
    elif isinstance(data, cudf.Series):
        return _sparsify_and_convert(data.to_numpy(), input_type)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return _sparsify_and_convert(data.to_numpy(), input_type)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))


def _convert_to_scipy_sparse_csr(data):
    return _convert_to_scipy_sparse(data, "csr")


def _convert_to_scipy_sparse_csc(data):
    return _convert_to_scipy_sparse(data, "csc")


_data_generators = {
    "blobs": _gen_data_blobs,
    "zeros": _gen_data_zeros,
    "classification": _gen_data_classification,
    "regression": _gen_data_regression,
    "airline_regression": _gen_data_airline_regression,
    "airline_classification": _gen_data_airline_classification,
    "bosch": _gen_data_bosch,
    "covtype": _gen_data_covtype,
    "epsilon": _gen_data_epsilon,
    "fraud": _gen_data_fraud,
    "higgs": _gen_data_higgs,
    "year": _gen_data_year,
}

_data_converters = {
    "numpy": _convert_to_numpy,
    "cupy": _convert_to_cupy,
    "cudf": _convert_to_cudf,
    "pandas": _convert_to_pandas,
    "gpuarray": _convert_to_gpuarray,
    "gpuarray-c": _convert_to_gpuarray_c,
    "scipy-sparse-csr": _convert_to_scipy_sparse_csr,
    "scipy-sparse-csc": _convert_to_scipy_sparse_csc,
}


def all_datasets():
    return _data_generators


@functools.lru_cache(maxsize=8)
def gen_data(
    dataset_name,
    dataset_format,
    n_samples=0,
    n_features=0,
    test_fraction=0.0,
    datasets_root_dir=DATASETS_DIRECTORY,
    dtype=np.float32,
    **kwargs,
):
    """Returns a tuple of data from the specified generator.

    Parameters
    ----------
    dataset_name : str
        Dataset to use. Can be a synthetic generator (blobs or regression)
        or a specified dataset (higgs currently, others coming soon)
    dataset_format : str
        Type of data to return. (One of cudf, numpy, pandas, gpuarray)
    n_samples : int
        Total number of samples to loaded including training and testing samples
    test_fraction : float
        Fraction of the dataset to partition randomly into the test set.
        If this is 0.0, no test set will be created.

    Returns
    -------
        (train_features, train_labels, test_features, test_labels) tuple
        containing matrices or dataframes of the requested format.
        test_features and test_labels may be None if no splitting was done.
    """

    pickle_x_file_url = os.path.join(
        datasets_root_dir, "%s_x.pkl" % dataset_name
    )
    pickle_y_file_url = os.path.join(
        datasets_root_dir, "%s_y.pkl" % dataset_name
    )

    mock_datasets = ["regression", "classification", "blobs", "zero"]
    if dataset_name in mock_datasets:
        X_df, y_df = _data_generators[dataset_name](
            n_samples=n_samples, n_features=n_features, dtype=dtype, **kwargs
        )
    else:
        if os.path.isfile(pickle_x_file_url):
            # loading data from cache
            X = pd.read_pickle(pickle_x_file_url)
            y = pd.read_pickle(pickle_y_file_url)
        else:
            X, y = _data_generators[dataset_name](datasets_root_dir, **kwargs)

            # cache the dataset for future use
            X.to_pickle(pickle_x_file_url)
            y.to_pickle(pickle_y_file_url)

        if n_samples > X.shape[0]:
            raise ValueError(
                "%s dataset has only %d rows, cannot support %d"
                % (dataset_name, X.shape[0], n_samples)
            )

        if n_features > X.shape[1]:
            raise ValueError(
                "%s dataset has only %d features, cannot support %d"
                % (dataset_name, X.shape[1], n_features)
            )

        if n_samples == 0:
            n_samples = X.shape[0]

        if n_features == 0:
            n_features = X.shape[1]

        X_df = cudf.DataFrame(X.iloc[0:n_samples, 0:n_features].astype(dtype))
        y_df = cudf.Series(y.iloc[0:n_samples].astype(dtype))

    data = (X_df, y_df)
    if test_fraction != 0.0:
        random_state_dict = (
            {"random_state": kwargs["random_state"]}
            if "random_state" in kwargs
            else {}
        )
        X_train, X_test, y_train, y_test = tuple(
            sklearn.model_selection.train_test_split(
                *data,
                test_size=int(n_samples * test_fraction),
                **random_state_dict,
            )
        )
        data = (X_train, y_train, X_test, y_test)
    else:
        data = (*data, None, None)  # No test set
    data = _data_converters[dataset_format](data)
    return data
