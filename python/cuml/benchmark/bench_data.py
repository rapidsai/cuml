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
"""bench_data - Data generators for cuML benchmarks

The main entry point for consumers is generating

Each generator is a function that accepts:
 * n_samples (set to 0 for 'default')
 * n_features (set to 0 for 'default')
 * random_state
 * .. and optional generator-specific parameters

The function should return a 2-tuple (X, y), where X is a Pandas
dataframe and y is a Pandas series. If the generator does not produce
labels, it can return (X, None)

A set of helper functions (convert_*) can convert these to alternative
formats. Future revisions may support generating cudf dataframes or
GPU arrays directly instead.

"""

import numpy as np
import pandas as pd
import cudf
import os
import time
import pickle
import sklearn.datasets
from numba import cuda
from urllib.request import urlretrieve
import gzip

def _gen_data_regression(n_samples, n_features, random_state=42):
    """Wrapper for sklearn make_regression"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_features = 100
    X_arr, y_arr = sklearn.datasets.make_regression(n_samples, n_features, random_state=random_state)
    print("Xarr size: ", X_arr.shape)
    return (pd.DataFrame(X_arr.astype(np.float32)), pd.Series(y_arr.astype(np.float32)))

def _gen_data_blobs(n_samples, n_features, random_state=42, centers=None):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100
    X_arr, y_arr = sklearn.datasets.make_blobs(n_samples, n_features, centers=centers, random_state=random_state)
    return (pd.DataFrame(X_arr.astype(np.float32)), pd.Series(y_arr.astype(np.float32)))

def _gen_data_higgs(n_samples=None, n_features=None, random_state=42):
    X_df, y_df = load_higgs('pandas')
    if n_samples == 0:
        n_samples = X_df.shape[0]
    if n_features == 0:
        n_features = X_df.shape[1]
    if n_features > X_df.shape[1]:
        raise ValueError("Higgs dataset has only %d features, cannot support %d" %
                         (X_df.shape[1], n_features))
    if n_samples > X_df.shape[0]:
        raise ValueError("Higgs dataset has only %d rows, cannot support %d" %
                         (X_df.shape[0], n_samples))
    return X_df.iloc[:n_samples, :n_features], y_df.iloc[:n_samples]

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

def load_higgs(format='cudf'):
    """Returns the Higgs Boson dataset as an X, y tuple of dataframes."""
    higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    decompressed_filepath = _download_and_cache(higgs_url,
                                                os.path.join(DATASETS_DIRECTORY, "HIGGS.csv.gz"),
                                                os.path.join(DATASETS_DIRECTORY, "HIGGS.csv"))
    col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
    dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
    data_df = pd.read_csv(decompressed_filepath, names=col_names, dtype=dtypes_ls)
    X_df = data_df[data_df.columns.difference(['label'])]
    y_df = data_df['label']
    return X_df, y_df

def _convert_to_numpy(data):
    if isinstance(data, tuple):
        return tuple([_convert_to_numpy(d) for d in data])
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise Exception("Unsupported type %s" % str(type(data)))

def _convert_to_cudf(data):
    if isinstance(data, tuple):
        return tuple([_convert_to_cudf(d) for d in data])
    elif isinstance(data, pd.DataFrame):
        return cudf.DataFrame.from_pandas(data)
    elif isinstance(data, pd.Series):
        return cudf.Series.from_pandas(data)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))

def _convert_to_pandas(data):
    if isinstance(data, tuple):
        return tuple([_convert_to_pandas(d) for d in data])
    elif isinstance(data, pd.DataFrame):
        return cudf.DataFrame.from_pandas(data)
    elif isinstance(data, pd.Series):
        return cudf.Series.from_pandas(data)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))

def _convert_to_gpuarray(data, order='F'):
    if isinstance(data, tuple):
        return tuple([_convert_to_gpuarray(d, order=order)
                     for d in data])
    else:
        return input_to_dev_array(data, order=order)


_data_generators = { 'blobs': _gen_data_blobs,
                     'regression': _gen_data_regression,
                     'higgs': _gen_data_higgs }
_data_converters = { 'numpy': _convert_to_numpy,
                     'cudf': _convert_to_cudf,
                     'pandas': _convert_to_pandas,
                     'gpuarray': _convert_to_gpuarray }

def gen_data(dataset_name, dataset_format,
             n_samples=0, n_features=0, random_state=42,
             **kwargs):
    """Returns a tuple of data from the specified generator.

    Output
    -------
        (features, labels) tuple containing matrices or dataframes of the
        requested format.

    Parameters
    ----------
    dataset_name : str
        Dataset to use. Can be a synthetic generator (blobs or regression)
        or a specified dataset (higgs currently, others coming soon)

    dataset_format : str
        Type of data to return. (One of cudf, numpy, pandas, gpuarray)
    """
    data = _data_generators[dataset_name](n_samples, n_features, random_state, **kwargs)

    data = _data_converters[dataset_format](data)
    return data
