#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import numbers

import cupy as cp
import cupyx
import scipy
from cuml import Base
from cuml.common import input_to_cuml_array
from cuml.common.input_utils import input_to_cupy_array
import cuml.internals
from cuml.common.array import CumlArray
from cuml.decomposition import PCA


class IncrementalPCA(PCA):
    """
    Based on sklearn.decomposition.IncrementalPCA from scikit-learn 0.23.1

    Incremental principal components analysis (IPCA).
    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is
    centered but not scaled for each feature before applying the SVD.
    Depending on the size of the input data, this algorithm can be much
    more memory efficient than a PCA, and allows sparse input.
    This algorithm has constant memory complexity, on the order of
    :py:`batch_size * n_features`, enabling use of np.memmap files without
    loading the entire file into memory. For sparse matrices, the input
    is converted to dense in batches (in order to be able to subtract the
    mean) which avoids storing the entire dense matrix at any one time.
    The computational overhead of each SVD is
    :py:`O(batch_size * n_features ** 2)`, but only 2 * batch_size samples
    remain in memory at a time. There will be :py:`n_samples / batch_size`
    SVD computations to get the principal components, versus 1 large SVD
    of complexity :py:`O(n_samples * n_features ** 2)` for PCA.

    Parameters
    ----------

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    n_components : int or None, (default=None)
        Number of components to keep. If `n_components` is ``None``,
        then `n_components` is set to :py:`min(n_samples, n_features)`.
    whiten : bool, optional
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.
    copy : bool, (default=True)
        If False, X will be overwritten. :py:`copy=False` can be used to
        save memory but is unsafe for general use.
    batch_size : int or None, (default=None)
        The number of samples to use for each batch. Only used when calling
        `fit`. If `batch_size` is ``None``, then `batch_size`
        is inferred from the data and set to :py:`5 * n_features`, to provide a
        balance between approximation accuracy and memory consumption.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------

    components_ : array, shape (n_components, n_features)
        Components with maximum variance.
    explained_variance_ : array, shape (n_components,)
        Variance explained by each of the selected components.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If all components are stored, the sum of explained variances is equal
        to 1.0.
    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the `n_components`
        variables in the lower-dimensional space.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, aggregate over calls to `partial_fit`.
    var_ : array, shape (n_features,)
        Per-feature empirical variance, aggregate over calls to
        `partial_fit`.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from [4]_.
    n_components_ : int
        The estimated number of components. Relevant when
        `n_components=None`.
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across `partial_fit` calls.
    batch_size_ : int
        Inferred batch size from `batch_size`.

    Notes
    -----

    Implements the incremental PCA model from [1]_. This model is an extension
    of the Sequential Karhunen-Loeve Transform from [2]_. We have specifically
    abstained from an optimization used by authors of both papers, a QR
    decomposition used in specific situations to reduce the algorithmic
    complexity of the SVD. The source for this technique is [3]_. This
    technique has been omitted because it is advantageous only when decomposing
    a matrix with :py:`n_samples >= 5/3 * n_features` where `n_samples` and
    `n_features` are the matrix rows and columns, respectively. In addition,
    it hurts the readability of the implemented algorithm. This would be a good
    opportunity for future optimization, if it is deemed necessary.

    References
    ----------
    .. [1] `D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust
        Visual Tracking, International Journal of Computer Vision, Volume 77,
        Issue 1-3, pp. 125-141, May 2008.
        <https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf>`_

    .. [2] `A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis
        Extraction and its Application to Images, IEEE Transactions on Image
        Processing, Volume 9, Number 8, pp. 1371-1374, August 2000.
        <https://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf>`_

    .. [3] G. Golub and C. Van Loan. Matrix Computations, Third Edition,
        Chapter 5, Section 5.4.4, pp. 252-253.

    .. [4] `C. Bishop, 1999. "Pattern Recognition and Machine Learning",
        Section 12.2.1, pp. 574
        <http://www.miketipping.com/papers/met-mppca.pdf>`_

    Examples
    ---------

    .. code-block:: python

        >>> from cuml.decomposition import IncrementalPCA
        >>> import cupy as cp
        >>> import cupyx
        >>>
        >>> X = cupyx.scipy.sparse.random(1000, 4, format='csr', density=0.07)
        >>> ipca = IncrementalPCA(n_components=2, batch_size=200)
        >>> ipca.fit(X)
        >>>
        >>> # Components:
        >>> ipca.components_
        array([[-0.02362926,  0.87328851, -0.15971988,  0.45967206],
            [-0.14643883,  0.11414225,  0.97589354,  0.11471273]])
        >>>
        >>> # Singular Values:
        >>> ipca.singular_values_
        array([4.90298662, 4.54498226])
        >>>
        >>> # Explained Variance:
        >>> ipca.explained_variance_
        array([0.02406334, 0.02067754])
        >>>
        >>> # Explained Variance Ratio:
        >>> ipca.explained_variance_ratio_
        array([0.28018011, 0.24075775])
        >>>
        >>> # Mean:
        >>> ipca.mean_
        array([0.03249896, 0.03629852, 0.03268694, 0.03216601])
        >>>
        >>> # Noise Variance:
        >>> ipca.noise_variance_.item()
        0.003474966583315544

    """
    def __init__(self, handle=None, n_components=None, *, whiten=False,
                 copy=True, batch_size=None, verbose=False,
                 output_type=None):

        super(IncrementalPCA, self).__init__(handle=handle,
                                             n_components=n_components,
                                             whiten=whiten, copy=copy,
                                             verbose=verbose,
                                             output_type=output_type)
        self.batch_size = batch_size
        self._hyperparams = ["n_components", "whiten", "copy", "batch_size"]
        self._sparse_model = True

    def fit(self, X, y=None) -> "IncrementalPCA":
        """
        Fit the model with X, using minibatches of size batch_size.

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
        self.n_samples_seen_ = 0
        self.mean_ = .0
        self.var_ = .0

        if scipy.sparse.issparse(X) or cupyx.scipy.sparse.issparse(X):
            X = _validate_sparse_input(X)
        else:
            # NOTE: While we cast the input to a cupy array here, we still
            # respect the `output_type` parameter in the constructor. This
            # is done by PCA, which IncrementalPCA inherits from. PCA's
            # transform and inverse transform convert the output to the
            # required type.
            X, n_samples, n_features, self.dtype = \
                input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in _gen_batches(n_samples, self.batch_size_,
                                  min_batch_size=self.n_components or 0):
            X_batch = X[batch]
            if cupyx.scipy.sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            self.partial_fit(X_batch, check_input=False)

        return self

    @cuml.internals.api_base_return_any_skipall
    def partial_fit(self, X, y=None, check_input=True) -> "IncrementalPCA":
        """
        Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------

        X : array-like or sparse matrix, shape (n_samples, n_features)
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
            if scipy.sparse.issparse(X) or cupyx.scipy.sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches.")

            self._set_output_type(X)

            X, n_samples, n_features, self.dtype = \
                input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
        else:
            n_samples, n_features = X.shape

        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
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

        if (self.components_ is not None) and (self.components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self.components_.shape[0], self.n_components_))
        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
            self.var_ = .0

        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = \
            _incremental_mean_and_var(
                X, last_mean=self.mean_, last_variance=self.var_,
                last_sample_count=cp.repeat(cp.asarray([self.n_samples_seen_]),
                                            X.shape[1]))
        n_total_samples = n_total_samples[0]

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X = X - col_mean
        else:
            col_batch_mean = cp.mean(X, axis=0)
            X = X - col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                cp.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self.mean_ - col_batch_mean)
            X = cp.vstack((self.singular_values_.reshape((-1, 1)) *
                           self.components_, X, mean_correction))

        U, S, V = cp.linalg.svd(X, full_matrices=False)
        U, V = _svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        explained_variance_ratio = S ** 2 / cp.sum(col_var * n_total_samples)

        self.n_rows = n_total_samples
        self.n_samples_seen_ = n_total_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self.noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = 0.

        return self

    def transform(self, X, convert_dtype=False) -> CumlArray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set, using minibatches of size batch_size if X is
        sparse.

        Parameters
        ----------

        X : array-like or sparse matrix, shape (n_samples, n_features)
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

        if scipy.sparse.issparse(X) or cupyx.scipy.sparse.issparse(X):

            X = _validate_sparse_input(X)

            n_samples = X.shape[0]
            output = []
            for batch in _gen_batches(n_samples, self.batch_size_,
                                      min_batch_size=self.n_components or 0):
                output.append(super().transform(X[batch]))
            output, _, _, _ = \
                input_to_cuml_array(cp.vstack(output), order='K')

            return output
        else:
            return super().transform(X)

    def get_param_names(self):
        # Skip super() since we dont pass any extra parameters in __init__
        return Base.get_param_names(self) + self._hyperparams


def _validate_sparse_input(X):
    """
    Validate the format and dtype of sparse inputs.
    This function throws an error for any cupyx.scipy.sparse object that is not
    of type cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.csc_matrix.
    It also validates the dtype of the input to be 'float32' or 'float64'

    Parameters
    ----------

    X : scipy.sparse or cupyx.scipy.sparse object
        A sparse input

    Returns
    -------

    X : The input converted to a cupyx.scipy.sparse.csr_matrix object

    """

    acceptable_dtypes = ('float32', 'float64')

    # NOTE: We can include cupyx.scipy.sparse.csc.csc_matrix
    # once it supports indexing in cupy 8.0.0b5
    acceptable_cupy_sparse_formats = \
        (cupyx.scipy.sparse.csr_matrix)

    if X.dtype not in acceptable_dtypes:
        raise TypeError("Expected input to be of type float32 or float64."
                        " Received %s" % X.dtype)
    if scipy.sparse.issparse(X):
        return cupyx.scipy.sparse.csr_matrix(X)
    elif cupyx.scipy.sparse.issparse(X):
        if not isinstance(X, acceptable_cupy_sparse_formats):
            raise TypeError("Expected input to be of type"
                            " cupyx.scipy.sparse.csr_matrix or"
                            " cupyx.scipy.sparse.csc_matrix. Received %s"
                            % type(X))
        else:
            return X


def _gen_batches(n, batch_size, min_batch_size=0):
    """
    Generator to create slices containing batch_size elements, from 0 to n.
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
        result = op(x, *args, **kwargs, dtype=cp.float64).astype(cp.float32)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """
    Calculate mean update and a Youngs and Cramer variance update.
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

        # NOTE: The scikit-learn implementation has a np.errstate check
        # here for ignoring invalid divides. This is not implemented in
        # cupy as of 7.6.0
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
    """
    Sign correction to ensure deterministic output from SVD.
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
        signs = cp.sign(u[max_abs_cols, list(range(u.shape[1]))])
        u *= signs
        v *= signs[:, cp.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = cp.argmax(cp.abs(v), axis=1)
        signs = cp.sign(v[list(range(v.shape[0])), max_abs_rows])
        u *= signs
        v *= signs[:, cp.newaxis]
    return u, v
