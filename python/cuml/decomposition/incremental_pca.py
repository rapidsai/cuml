#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cupyx
import scipy
import numbers
import numpy as np

from cuml.common import with_cupy_rmm
from cuml.common.input_utils import sparse_scipy_to_cp
from cuml.common import input_to_cuml_array

from cuml.decomposition import PCA

class IncrementalPCA(PCA):
    def __init__(self, handle=None, n_components=None, *, whiten=False,
                 copy=True, batch_size=None, verbose=None, 
                 output_type=None):
        super(IncrementalPCA, self).__init__(handle=handle, n_components=n_components,
                                  whiten=whiten, copy=copy, verbose=verbose,
                                  output_type=output_type)
        self.batch_size = batch_size
        self._param_names = ["n_components", "whiten", "copy", "batch_size"]

    @with_cupy_rmm
    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._set_output_type(X)

        self._sparse_model = True

        self.n_samples_seen_ = 0
        self._mean_ = .0
        self.var_ = .0
        
        if scipy.sparse.issparse(X):
            X = sparse_scipy_to_cp(X)
        elif cp.sparse.issparse(X):
            pass
        else:
            X, n_samples, n_features, self.dtype = \
                input_to_cuml_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
            X = X.to_output(output_type='cupy')
        
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in _gen_batches(n_samples, self.batch_size_,
                                  min_batch_size=self.n_components or 0):
            X_batch = X[batch]
            if cp.sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            self.partial_fit(X_batch, check_input=False)

        return self

    @with_cupy_rmm
    def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        check_input : bool
            Run check_array on X.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if check_input:
            if scipy.sparse.issparse(X) or cp.sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches.")
            
            self._set_output_type(X)

            X, n_samples, n_features, self.dtype = \
                input_to_cuml_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
            X = X.to_output(output_type='cupy')
        else:
            n_samples, n_features = X.shape

        if not hasattr(self, '_components_'):
            self._components_ = None

        if self.n_components is None:
            if self._components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self._components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self._components_ is not None) and (self._components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self._components_.shape[0], self.n_components_))

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self._mean_ = .0
            self.var_ = .0

        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = \
            _incremental_mean_and_var(
                X, last_mean=self._mean_, last_variance=self.var_,
                last_sample_count=cp.asarray(np.repeat(self.n_samples_seen_, X.shape[1])))
        n_total_samples = n_total_samples[0]

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = cp.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                cp.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self._mean_ - col_batch_mean)
            X = cp.vstack((self.singular_values_.reshape((-1, 1)) *
                           self.components_, X, mean_correction))

        U, S, V = cp.linalg.svd(X, full_matrices=False)
        U, V = _svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        explained_variance_ratio = S ** 2 / cp.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self._components_ = V[:self.n_components_]
        self._singular_values_ = S[:self.n_components_]
        self._mean_ = col_mean
        self.var_ = col_var
        self._explained_variance_ = explained_variance[:self.n_components_]
        self._explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self._noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self._noise_variance_ = 0.
        return self
    
    @with_cupy_rmm
    def transform(self, X, convert_dtype=False):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set, using minibatches of size batch_size if X is
        sparse.
        Parameters
        ----------
        X : array-like (device or host), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        
        convert_dtype : bool, optional (default = False)
            When set to True, the transform method will automatically
            convert the input to the data type which was used to train the
            model. This will increase memory used for the method.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        if scipy.sparse.issparse(X) or cp.sparse.issparse(X):
            out_type = self._get_output_type(X)

            if scipy.sparse.issparse(X):
                X = sparse_scipy_to_cp(X)

            n_samples = X.shape[0]
            output = []
            for batch in _gen_batches(n_samples, self.batch_size_,
                                    min_batch_size=self.n_components or 0):
                output.append(super().transform(X[batch]))
            output = cp.vstack(output)

            return output.to_output(out_type=out_type)
        else:
            return super().transform(X)

    def get_param_names(self):
        return self._param_names


def _gen_batches(n, batch_size, min_batch_size=0):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.

    Parameters
    ----------
    n : int
    batch_size : int
        Number of element in each batch
    min_batch_size : int, default=0
        Minimum batch size to produce.
    Yields
    ------
    slice of batch_size elements
    """

    if not isinstance(batch_size, numbers.Integral):
        raise TypeError("gen_batches got batch_size=%s, must be an"
                        " integer" % batch_size)
    if batch_size <= 0:
        raise ValueError("gen_batches got batch_size=%s, must be"
                         " positive" % batch_size)
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A cupy accumulator function such as cp.mean or cp.sum
    x : cupy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function
    Returns
    -------
    result : The output of the accumulator function passed to this function
    """

    if cp.issubdtype(x.dtype, cp.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=cp.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : array-like, shape (n_features,)
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : array, shape (n_features,)
    Notes
    -----
    NaNs are ignored during the algorithm.
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    """

    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = _safe_accumulator_op(cp.nansum, X, axis=0)

    new_sample_count = cp.sum(~cp.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (
            _safe_accumulator_op(cp.nanvar, X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance + new_unnormalized_variance +
            last_over_new_count / updated_sample_count *
            (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def _svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : cupy.ndarray
        u and v are the output of `cupy.linalg.svd`
    v : cupy.ndarray
        u and v are the output of `cupy.linalg.svd`
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = cp.argmax(cp.abs(u), axis=0)
        signs = cp.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, cp.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = cp.argmax(cp.abs(v), axis=1)
        signs = cp.sign(v[list(range(v.shape[0])), max_abs_rows])
        u *= signs
        v *= signs[:, cp.newaxis]
    return u, v
