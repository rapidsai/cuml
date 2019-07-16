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

Each generator is a function that accepts:
 * n_samples (set to 0 for 'default')
 * n_features (set to 0 for 'default')
 * random_state
 * .. and optional generator-specific parameters

The function should return a 2-tuple (X, y), where X is a Pandas
dataframe and y is a Pandas series. If the generator does not produce
labels, it can return (X, None)

Future revisions will support generating cudf dataframes or GPU
arrays instead.

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

def gen_data_regression(n_samples, n_features, random_state=42):
    """Wrapper for sklearn make_regression"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_features = 100
    X_arr, y_arr = sklearn.datasets.make_regression(n_samples, n_features, random_state=random_state)
    print("Xarr size: ", X_arr.shape)
    return (pd.DataFrame(X_arr), pd.Series(y_arr))

def gen_data_blobs(n_samples, n_features, random_state=42, centers=None):
    """Wrapper for sklearn make_blobs"""
    if n_samples == 0:
        n_samples = int(1e6)
    if n_features == 0:
        n_samples = 100
    X_arr, y_arr = sklearn.datasets.make_blobs(n_samples, n_features, centers=centers, random_state=random_state)
    return (pd.DataFrame(X_arr), pd.Series(y_arr))

def gen_data_higgs(n_samples=None, n_features=None, random_state=42):
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
    data_cudf = cudf.read_csv(decompressed_filepath, names=col_names, dtype=dtypes_ls)
    X_cudf = data_cudf[data_cudf.columns.difference(['label'])]
    y_cudf = data_cudf['label']
    if format == 'cudf':
        return X_cudf, y_cudf
    elif format == 'gpuarray':
        return X_cudf.as_matrix(), y_cudf.to_array()
    elif format == 'numpy':
        return X_cudf.as_matrix().as_numpy(), y_cudf.to_array().as_numpy()
    elif format == 'pandas':
        return X_cudf.to_pandas(), y_cudf.to_pandas()
    else:
        raise ValueError("Unknown format: %s" % format)


