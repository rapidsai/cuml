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

import cudf
import gzip
import functools
import os
import numpy as np
import cupy as cp
import pandas as pd

import cuml.datasets
import sklearn.model_selection

from urllib.request import urlretrieve
from cuml.internals import input_utils
from numba import cuda

from cuml.internals.import_utils import has_scipy


def _gen_data_regression(n_samples, n_features, random_state=42,
                         dtype=np.float32):
    """Wrapper for sklearn make_regression"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_features = 100

    X_arr, y_arr = cuml.datasets.make_regression(
        n_samples=n_samples, n_features=n_features,
        random_state=random_state, dtype=dtype)

    return X_arr, y_arr


def _gen_data_blobs(n_samples, n_features, random_state=42, centers=None,
                    dtype=np.float32):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100

    X_arr, y_arr = cuml.datasets.make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers,
        random_state=random_state, dtype=dtype)

    return X_arr, y_arr


def _gen_data_zeros(n_samples, n_features, dtype=np.float32):
    """Dummy generator for use in testing - returns all 0s"""
    return cp.zeros((n_samples, n_features), dtype=dtype), \
        cp.zeros(n_samples, dtype=dtype)


def _gen_data_classification(n_samples, n_features, random_state=42,
                             n_classes=2, dtype=np.float32):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100

    X_arr, y_arr = cuml.datasets.make_classification(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes,
        random_state=random_state, dtype=dtype)

    return X_arr, y_arr


def _gen_data_higgs(n_samples=None, n_features=None, dtype=np.float32):
    """Wrapper returning Higgs in Pandas format"""
    X_df, y_df = load_higgs()
    if n_samples == 0:
        n_samples = X_df.shape[0]
    if n_features == 0:
        n_features = X_df.shape[1]
    if n_features > X_df.shape[1]:
        raise ValueError(
            "Higgs dataset has only %d features, cannot support %d"
            % (X_df.shape[1], n_features)
        )
    if n_samples > X_df.shape[0]:
        raise ValueError(
            "Higgs dataset has only %d rows, cannot support %d"
            % (X_df.shape[0], n_samples)
        )
    return X_df.iloc[:n_samples, :n_features].astype(dtype), \
        y_df.iloc[:n_samples].astype(dtype)


def _download_and_cache(url, compressed_filepath, decompressed_filepath):
    if not os.path.isfile(compressed_filepath):
        urlretrieve(url, compressed_filepath)
    if not os.path.isfile(decompressed_filepath):
        cf = gzip.GzipFile(compressed_filepath)
        with open(decompressed_filepath, 'wb') as df:
            df.write(cf.read())
    return decompressed_filepath


# Default location to cache datasets
DATASETS_DIRECTORY = '.'


def load_higgs():
    """Returns the Higgs Boson dataset as an X, y tuple of dataframes."""
    higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'  # noqa
    decompressed_filepath = _download_and_cache(
        higgs_url,
        os.path.join(DATASETS_DIRECTORY, "HIGGS.csv.gz"),
        os.path.join(DATASETS_DIRECTORY, "HIGGS.csv"),
    )
    col_names = ['label'] + [
        "col-{}".format(i) for i in range(2, 30)
    ]  # Assign column names
    dtypes_ls = [np.int32] + [
        np.float32 for _ in range(2, 30)
    ]  # Assign dtypes to each column
    data_df = pd.read_csv(
        decompressed_filepath, names=col_names,
        dtype={k: v for k, v in zip(col_names, dtypes_ls)}
    )
    X_df = data_df[data_df.columns.difference(['label'])]
    y_df = data_df['label']
    return cudf.DataFrame.from_pandas(X_df), cudf.Series.from_pandas(y_df)


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
        return cudf.DataFrame.from_pandas(data)
    elif isinstance(data, pd.Series):
        return cudf.Series.from_pandas(data)
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


def _convert_to_gpuarray(data, order='F'):
    if data is None:
        return None
    elif isinstance(data, tuple):
        return tuple([_convert_to_gpuarray(d, order=order) for d in data])
    elif isinstance(data, pd.DataFrame):
        return _convert_to_gpuarray(cudf.DataFrame.from_pandas(data),
                                    order=order)
    elif isinstance(data, pd.Series):
        gs = cudf.Series.from_pandas(data)
        return cuda.as_cuda_array(gs)
    else:
        return input_utils.input_to_cuml_array(
            data, order=order)[0].to_output("numba")


def _convert_to_gpuarray_c(data):
    return _convert_to_gpuarray(data, order='C')


def _sparsify_and_convert(data, input_type, sparsity_ratio=0.3):
    """Randomly set values to 0 and produce a sparse array."""
    if not has_scipy():
        raise RuntimeError("Scipy is required")
    import scipy
    random_loc = np.random.choice(data.size,
                                  int(data.size * sparsity_ratio),
                                  replace=False)
    data.ravel()[random_loc] = 0
    if input_type == 'csr':
        return scipy.sparse.csr_matrix(data)
    elif input_type == 'csc':
        return scipy.sparse.csc_matrix(data)
    else:
        TypeError('Wrong sparse input type {}'.format(input_type))


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
    return _convert_to_scipy_sparse(data, 'csr')


def _convert_to_scipy_sparse_csc(data):
    return _convert_to_scipy_sparse(data, 'csc')


_data_generators = {
    'blobs': _gen_data_blobs,
    'zeros': _gen_data_zeros,
    'classification': _gen_data_classification,
    'regression': _gen_data_regression,
    'higgs': _gen_data_higgs
}
_data_converters = {
    'numpy': _convert_to_numpy,
    'cupy': _convert_to_cupy,
    'cudf': _convert_to_cudf,
    'pandas': _convert_to_pandas,
    'gpuarray': _convert_to_gpuarray,
    'gpuarray-c': _convert_to_gpuarray_c,
    'scipy-sparse-csr': _convert_to_scipy_sparse_csr,
    'scipy-sparse-csc': _convert_to_scipy_sparse_csc
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
    **kwargs
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
        Number of samples to include in training set (regardless of test split)
    test_fraction : float
        Fraction of the dataset to partition randomly into the test set.
        If this is 0.0, no test set will be created.

    Returns
    -------
        (train_features, train_labels, test_features, test_labels) tuple
        containing matrices or dataframes of the requested format.
        test_features and test_labels may be None if no splitting was done.
    """
    data = _data_generators[dataset_name](
        int(n_samples / (1 - test_fraction)),
        n_features,
        **kwargs
    )
    if test_fraction != 0.0:
        if n_samples == 0:
            n_samples = int(data[0].shape[0] * (1 - test_fraction))
        random_state_dict = ({'random_state': kwargs['random_state']}
                             if 'random_state' in kwargs else {})
        X_train, X_test, y_train, y_test = tuple(
            sklearn.model_selection.train_test_split(
                *data, train_size=n_samples,
                **random_state_dict
            )
        )
        data = (X_train, y_train, X_test, y_test)
    else:
        data = (*data, None, None)  # No test set

    data = _data_converters[dataset_format](data)
    return data
