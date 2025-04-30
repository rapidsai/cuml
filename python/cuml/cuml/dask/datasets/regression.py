#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import cupy as cp
import dask.array as da
import numpy as np

from cuml.common import with_cupy_rmm
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import get_client
from cuml.dask.datasets.utils import _create_delayed, _get_labels, _get_X


def _create_rs_generator(random_state):
    if hasattr(random_state, "__module__"):
        rs_type = random_state.__module__ + "." + type(random_state).__name__
    else:
        rs_type = type(random_state).__name__

    rs = None
    if rs_type == "NoneType" or rs_type == "int":
        rs = da.random.RandomState(
            seed=random_state, RandomState=cp.random.RandomState
        )
    elif rs_type == "cupy.random.generator.RandomState":
        rs = da.random.RandomState(RandomState=random_state)
    elif rs_type == "dask.array.random.RandomState":
        rs = random_state
    else:
        raise ValueError(
            "random_state type must be int, CuPy RandomState \
                          or Dask RandomState"
        )
    return rs


def _dask_f_order_standard_normal(nrows, ncols, dtype, seed):
    local_rs = cp.random.RandomState(seed=seed)
    x = local_rs.standard_normal(nrows * ncols, dtype=dtype)
    x = x.reshape((nrows, ncols), order="F")
    return x


def _f_order_standard_normal(client, rs, chunksizes, ncols, dtype):
    workers = list(client.has_what().keys())

    n_chunks = len(chunksizes)
    chunks_workers = (workers * n_chunks)[:n_chunks]

    chunk_seeds = rs.permutation(len(chunksizes))
    chunks = [
        client.submit(
            _dask_f_order_standard_normal,
            chunksize,
            ncols,
            dtype,
            chunk_seeds[idx],
            workers=[chunks_workers[idx]],
            pure=False,
        )
        for idx, chunksize in enumerate(chunksizes)
    ]

    chunks_dela = _create_delayed(chunks, dtype, chunksizes, ncols)

    return da.concatenate(chunks_dela, axis=0)


def _dask_data_from_multivariate_normal(
    seed, covar, n_samples, n_features, dtype
):
    mean = cp.zeros(n_features)
    local_rs = cp.random.RandomState()
    return local_rs.multivariate_normal(mean, covar, n_samples, dtype=dtype)


def _data_from_multivariate_normal(
    client, rs, covar, chunksizes, n_features, dtype
):
    workers = list(client.has_what().keys())

    n_chunks = len(chunksizes)
    chunks_workers = (workers * n_chunks)[:n_chunks]

    chunk_seeds = rs.permutation(len(chunksizes))

    data_parts = [
        client.submit(
            _dask_data_from_multivariate_normal,
            chunk_seeds[idx],
            covar,
            chunksizes[idx],
            n_features,
            dtype,
            workers=[chunks_workers[idx]],
            pure=False,
        )
        for idx, chunk in enumerate(chunksizes)
    ]

    data_dela = _create_delayed(data_parts, dtype, chunksizes, n_features)

    return da.concatenate(data_dela, axis=0)


def _dask_shuffle(part, n_samples, seed, features_indices):
    X, y = part[0], part[1]
    local_rs = cp.random.RandomState(seed=seed)
    samples_indices = local_rs.permutation(n_samples)

    X[...] = X[samples_indices, :]
    X[...] = X[:, features_indices]

    y[...] = y[samples_indices, :]
    return X, y


def _shuffle(
    client,
    rs,
    X,
    y,
    chunksizes,
    n_features,
    features_indices,
    n_targets,
    dtype,
):
    data_ddh = DistributedDataHandler.create(data=(X, y), client=client)

    chunk_seeds = rs.permutation(len(chunksizes))

    shuffled = [
        client.submit(
            _dask_shuffle,
            part,
            chunksizes[idx],
            chunk_seeds[idx],
            features_indices,
            workers=[w],
            pure=False,
        )
        for idx, (w, part) in enumerate(data_ddh.gpu_futures)
    ]

    X_shuffled = [
        client.submit(_get_X, f, pure=False) for idx, f in enumerate(shuffled)
    ]
    y_shuffled = [
        client.submit(_get_labels, f, pure=False)
        for idx, f in enumerate(shuffled)
    ]

    X_dela = _create_delayed(X_shuffled, dtype, chunksizes, n_features)
    y_dela = _create_delayed(y_shuffled, dtype, chunksizes, n_targets)

    return da.concatenate(X_dela, axis=0), da.concatenate(y_dela, axis=0)


def _convert_to_order(client, X, chunksizes, order, n_features, dtype):
    X_ddh = DistributedDataHandler.create(data=X, client=client)
    X_converted = [
        client.submit(cp.array, X_part, copy=False, order=order, workers=[w])
        for idx, (w, X_part) in enumerate(X_ddh.gpu_futures)
    ]

    X_dela = _create_delayed(X_converted, dtype, chunksizes, n_features)

    return da.concatenate(X_dela, axis=0)


def _generate_chunks_for_qr(total_size, min_size, n_parts):

    n_total_per_part = max(1, int(total_size / n_parts))
    if n_total_per_part > min_size:
        min_size = n_total_per_part

    n_partitions = int(max(1, total_size / min_size))
    rest = total_size % (n_partitions * min_size)
    chunks_list = [min_size for i in range(n_partitions - 1)]
    chunks_list.append(min_size + rest)
    return tuple(chunks_list)


def _generate_singular_values(
    n, effective_rank, tail_strength, n_samples_per_part, dtype="float32"
):
    # Index of the singular values
    sing_ind = cp.arange(n, dtype=dtype)

    # Build the singular profile by assembling signal and noise components
    tmp = sing_ind / effective_rank
    low_rank = (1 - tail_strength) * cp.exp(-1.0 * tmp**2)
    tail = tail_strength * cp.exp(-0.1 * tmp)
    s = low_rank + tail
    return s


def _dask_make_low_rank_covariance(
    n_features,
    effective_rank,
    tail_strength,
    seed,
    n_parts,
    n_samples_per_part,
    dtype,
):
    """
    This approach is a faster approach than making X as a full low
    rank matrix. Here, we take advantage of the fact that with
    SVD, X * X^T = V * S^2 * V^T. This means that we can
    generate a covariance matrix by generating only the right
    eigen-vector and the squared, low-rank singular values.
    With a memory usage of only O(n_features ^ 2) in this case, we pass
    this covariance matrix to workers to generate each part of X
    embarrassingly parallel from a multi-variate normal with mean 0
    and generated covariance.
    """
    local_rs = cp.random.RandomState(seed=seed)
    m2 = local_rs.standard_normal((n_features, n_features), dtype=dtype)
    v, _ = cp.linalg.qr(m2)

    s = _generate_singular_values(
        n_features, effective_rank, tail_strength, n_samples_per_part
    )

    v *= s**2
    return cp.dot(v, cp.transpose(v))


def _make_low_rank_covariance(
    client,
    n_features,
    effective_rank,
    tail_strength,
    seed,
    n_parts,
    n_samples_per_part,
    dtype,
):

    return client.submit(
        _dask_make_low_rank_covariance,
        n_features,
        effective_rank,
        tail_strength,
        seed,
        n_parts,
        n_samples_per_part,
        dtype,
    )


def make_low_rank_matrix(
    n_samples=100,
    n_features=100,
    effective_rank=10,
    tail_strength=0.5,
    random_state=None,
    n_parts=1,
    n_samples_per_part=None,
    dtype="float32",
):
    """ Generate a mostly low rank matrix with bell-shaped singular values

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=100)
        The number of features.
    effective_rank : int, optional (default=10)
        The approximate number of singular vectors required to explain most of
        the data by linear combinations.
    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile.
    random_state : int, CuPy RandomState instance, Dask RandomState instance \
                   or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    n_parts : int, optional (default=1)
        The number of parts of work.
    dtype: str, optional (default='float32')
        dtype of generated data

    Returns
    -------
    X : Dask-CuPy array of shape [n_samples, n_features]
        The matrix.

    """

    rs = _create_rs_generator(random_state)
    n = min(n_samples, n_features)

    # Random (ortho normal) vectors
    m1 = rs.standard_normal(
        (n_samples, n),
        chunks=(_generate_chunks_for_qr(n_samples, n, n_parts), -1),
        dtype=dtype,
    )
    u, _ = da.linalg.qr(m1)

    m2 = rs.standard_normal(
        (n, n_features),
        chunks=(-1, _generate_chunks_for_qr(n_features, n, n_parts)),
        dtype=dtype,
    )
    v, _ = da.linalg.qr(m2)

    # For final multiplication
    if n_samples_per_part is None:
        n_samples_per_part = max(1, int(n_samples / n_parts))
    u = u.rechunk({0: n_samples_per_part, 1: -1})
    v = v.rechunk({0: n_samples_per_part, 1: -1})

    local_s = _generate_singular_values(
        n, effective_rank, tail_strength, n_samples_per_part
    )
    s = da.from_array(local_s, chunks=(int(n_samples_per_part),))

    u *= s
    return da.dot(u, v)


@with_cupy_rmm
def make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=False,
    coef=False,
    random_state=None,
    n_parts=1,
    n_samples_per_part=None,
    order="F",
    dtype="float32",
    client=None,
    use_full_low_rank=True,
):
    """
    Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile.

    The output is generated by applying a (potentially biased) random linear
    regression model with "n_informative" nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=100)
        The number of features.
    n_informative : int, optional (default=10)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.
    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.
    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.
    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.

        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if "effective_rank" is not None.
    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.
    shuffle : boolean, optional (default=False)
        Shuffle the samples and the features.
    coef : boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.
    random_state : int, CuPy RandomState instance, Dask RandomState instance \
                   or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    n_parts : int, optional (default=1)
        The number of parts of work.
    order : str, optional (default='F')
        Row-major or Col-major
    dtype: str, optional (default='float32')
        dtype of generated data
    use_full_low_rank : boolean (default=True)
        Whether to use the entire dataset to generate the low rank matrix.
        If False, it creates a low rank covariance and uses the
        corresponding covariance to generate a multivariate normal
        distribution on the remaining chunks

    Returns
    -------
    X : Dask-CuPy array of shape [n_samples, n_features]
        The input samples.
    y : Dask-CuPy array of shape [n_samples] or [n_samples, n_targets]
        The output values.
    coef : Dask-CuPy array of shape [n_features] \
           or [n_features, n_targets], optional
        The coefficient of the underlying linear model. It is returned only if
        coef is True.

    Notes
    -----
    Known Performance Limitations:
     1. When `effective_rank` is set and `use_full_low_rank` is True, \
        we cannot generate order `F` by construction, and an explicit \
        transpose is performed on each part. This may cause memory to spike \
        (other parameters make order `F` by construction)
     2. When `n_targets > 1` and `order = 'F'` as above, we have to \
        explicitly transpose the `y` array. If `coef = True`, then we also \
        explicitly transpose the `ground_truth` array
     3. When `shuffle = True` and `order = F`, there are memory spikes to \
        shuffle the `F` order arrays

    .. note:: If out-of-memory errors are encountered in any of the above
        configurations, try increasing the `n_parts` parameter.
    """

    client = get_client(client=client)

    n_informative = min(n_features, n_informative)
    rs = _create_rs_generator(random_state)

    if n_samples_per_part is None:
        n_samples_per_part = max(1, int(n_samples / n_parts))

    data_chunksizes = [n_samples_per_part] * n_parts

    data_chunksizes[-1] += n_samples % n_parts

    data_chunksizes = tuple(data_chunksizes)

    if effective_rank is None:
        # Randomly generate a well conditioned input set
        if order == "F":
            X = _f_order_standard_normal(
                client, rs, data_chunksizes, n_features, dtype
            )

        elif order == "C":
            X = rs.standard_normal(
                (n_samples, n_features),
                chunks=(data_chunksizes, -1),
                dtype=dtype,
            )

    else:
        # Randomly generate a low rank, fat tail input set
        if use_full_low_rank:
            X = make_low_rank_matrix(
                n_samples=n_samples,
                n_features=n_features,
                effective_rank=effective_rank,
                tail_strength=tail_strength,
                random_state=rs,
                n_parts=n_parts,
                n_samples_per_part=n_samples_per_part,
                dtype=dtype,
            )

            X = X.rechunk({0: data_chunksizes, 1: -1})
        else:
            seed = int(rs.randint(n_samples).compute())
            covar = _make_low_rank_covariance(
                client,
                n_features,
                effective_rank,
                tail_strength,
                seed,
                n_parts,
                n_samples_per_part,
                dtype,
            )
            X = _data_from_multivariate_normal(
                client, rs, covar, data_chunksizes, n_features, dtype
            )

        X = _convert_to_order(
            client, X, data_chunksizes, order, n_features, dtype
        )

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = 100.0 * rs.standard_normal(
        (n_informative, n_targets),
        chunks=(n_samples_per_part, -1),
        dtype=dtype,
    )

    y = da.dot(X[:, :n_informative], ground_truth) + bias

    if n_informative != n_features:
        zeroes = 0.0 * rs.standard_normal(
            (n_features - n_informative, n_targets), dtype=dtype
        )
        ground_truth = da.concatenate([ground_truth, zeroes], axis=0)

    ground_truth = ground_truth.rechunk(-1)

    # Add noise
    if noise > 0.0:
        y += rs.normal(scale=noise, size=y.shape, dtype=dtype)

    # Randomly permute samples and features
    if shuffle:
        features_indices = np.random.permutation(n_features)
        X, y = _shuffle(
            client,
            rs,
            X,
            y,
            data_chunksizes,
            n_features,
            features_indices,
            n_targets,
            dtype,
        )

        ground_truth = ground_truth[features_indices, :]

    y = da.squeeze(y)

    if order == "F" and n_targets > 1:
        y = _convert_to_order(client, y, y.chunks[0], order, n_targets, dtype)
        if coef:
            ground_truth = _convert_to_order(
                client,
                ground_truth,
                ground_truth.chunks[0],
                order,
                n_targets,
                dtype,
            )

    if coef:
        ground_truth = da.squeeze(ground_truth)
        return X, y, ground_truth

    else:
        return X, y
