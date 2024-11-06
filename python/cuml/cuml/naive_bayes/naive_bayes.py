#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.prims.array import binarize
from cuml.prims.label import invert_labels
from cuml.prims.label import check_labels
from cuml.prims.label import make_monotonic
from cuml.internals.import_utils import has_scipy
from cuml.common.doc_utils import generate_docstring
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.base import Base
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common import CumlArray
import math
import warnings
from cuml.internals.safe_imports import (
    gpu_only_import,
    gpu_only_import_from,
    null_decorator,
)

nvtx_annotate = gpu_only_import_from("nvtx", "annotate", alt=null_decorator)
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


def count_features_coo_kernel(float_dtype, int_dtype):
    """
    A simple reduction kernel that takes in a sparse (COO) array
    of features and computes the sum (or sum squared) for each class
    label
    """

    kernel_str = r"""({0} *out,
                    int *rows, int *cols,
                    {0} *vals, int nnz,
                    int n_rows, int n_cols,
                    {1} *labels,
                    {0} *weights,
                    bool has_weights,
                    int n_classes,
                    bool square) {

      int i = blockIdx.x * blockDim.x + threadIdx.x;

      if(i >= nnz) return;

      int row = rows[i];
      int col = cols[i];
      {0} val = vals[i];
      {1} label = labels[row];
      unsigned out_idx = (col * n_classes) + label;
      if(has_weights)
        val *= weights[i];

      if(square) val *= val;
      atomicAdd(out + out_idx, val);
    }"""

    return cuda_kernel_factory(
        kernel_str, (float_dtype, int_dtype), "count_features_coo"
    )


def count_classes_kernel(float_dtype, int_dtype):
    kernel_str = r"""
    ({0} *out, int n_rows, {1} *labels) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if(row >= n_rows) return;
      {1} label = labels[row];
      atomicAdd(out + label, ({0})1);
    }"""

    return cuda_kernel_factory(
        kernel_str, (float_dtype, int_dtype), "count_classes"
    )


def count_features_dense_kernel(float_dtype, int_dtype):

    kernel_str = r"""
    ({0} *out,
     {0} *in,
     int n_rows,
     int n_cols,
     {1} *labels,
     {0} *weights,
     bool has_weights,
     int n_classes,
     bool square,
     bool rowMajor,
     bool categorical) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;

      if(row >= n_rows || col >= n_cols) return;

      {0} val = !rowMajor ?
            in[col * n_rows + row] : in[row * n_cols + col];
      {1} label = labels[row];
      unsigned out_idx = ((col * n_classes) + label);

      if (categorical)
      {
        out_idx = (val * n_classes * n_cols) + (label * n_cols) + col;
        val = 1;
      }
      if(has_weights)
        val *= weights[row];

      if(val == 0.0) return;

      if(square) val *= val;

      atomicAdd(out + out_idx, val);
    }"""

    return cuda_kernel_factory(
        kernel_str, (float_dtype, int_dtype), "count_features_dense"
    )


def _convert_x_sparse(X):
    X = X.tocoo()

    if X.dtype not in [cp.float32, cp.float64]:
        raise ValueError(
            "Only floating-point dtypes (float32 or "
            "float64) are supported for sparse inputs."
        )

    rows = cp.asarray(X.row, dtype=X.row.dtype)
    cols = cp.asarray(X.col, dtype=X.col.dtype)
    data = cp.asarray(X.data, dtype=X.data.dtype)
    return cupyx.scipy.sparse.coo_matrix((data, (rows, cols)), shape=X.shape)


class _BaseNB(Base, ClassifierMixin):

    classes_ = CumlArrayDescriptor()
    class_count_ = CumlArrayDescriptor()
    feature_count_ = CumlArrayDescriptor()
    class_log_prior_ = CumlArrayDescriptor()
    feature_log_prob_ = CumlArrayDescriptor()

    def __init__(self, *, verbose=False, handle=None, output_type=None):
        super(_BaseNB, self).__init__(
            verbose=verbose, handle=handle, output_type=output_type
        )

    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks."""
        return X

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "y_hat",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_rows, 1)",
        },
    )
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Perform classification on an array of test vectors X.

        """
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.internals.import_utils import (
                dummy_function_always_false as scipy_sparse_isspmatrix,
            )

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
            index = None
        else:
            X = input_to_cuml_array(
                X,
                order="K",
                convert_to_dtype=(cp.float32 if convert_dtype else None),
                check_dtype=[cp.float32, cp.float64, cp.int32],
            )
            index = X.index
            # todo: improve index management for cupy based codebases
            X = X.array.to_output("cupy")

        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1).astype(self.classes_.dtype)

        y_hat = invert_labels(indices, classes=self.classes_)
        y_hat = CumlArray(data=y_hat, index=index)
        return y_hat

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "C",
            "type": "dense",
            "description": (
                "Returns the log-probability of the samples for each class in "
                "the model. The columns correspond to the classes in sorted "
                "order, as they appear in the attribute `classes_`."
            ),
            "shape": "(n_rows, 1)",
        },
    )
    def predict_log_proba(self, X, convert_dtype=True) -> CumlArray:
        """
        Return log-probability estimates for the test vector X.

        """
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.internals.import_utils import (
                dummy_function_always_false as scipy_sparse_isspmatrix,
            )

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
            index = None
        else:
            X = input_to_cuml_array(
                X,
                order="K",
                convert_to_dtype=(cp.float32 if convert_dtype else None),
                check_dtype=[cp.float32, cp.float64, cp.int32],
            )
            index = X.index
            # todo: improve index management for cupy based codebases
            X = X.array.to_output("cupy")

        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)

        # normalize by P(X) = P(f_1, ..., f_n)

        # Compute log(sum(exp()))

        # Subtract max in exp to prevent inf
        a_max = cp.amax(jll, axis=1, keepdims=True)

        exp = cp.exp(jll - a_max)
        logsumexp = cp.log(cp.sum(exp, axis=1))

        a_max = cp.squeeze(a_max, axis=1)

        log_prob_x = a_max + logsumexp

        if log_prob_x.ndim < 2:
            log_prob_x = log_prob_x.reshape((1, log_prob_x.shape[0]))
        result = jll - log_prob_x.T
        result = CumlArray(data=result, index=index)
        return result

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "C",
            "type": "dense",
            "description": (
                "Returns the probability of the samples for each class in the "
                "model. The columns correspond to the classes in sorted order,"
                " as they appear in the attribute `classes_`."
            ),
            "shape": "(n_rows, 1)",
        },
    )
    def predict_proba(self, X) -> CumlArray:
        """
        Return probability estimates for the test vector X.
        """
        result = cp.exp(self.predict_log_proba(X))
        return result


class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)
    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Parameters
    ----------
    priors : array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> X = cp.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1],
        ...                 [3, 2]], cp.float32)
        >>> Y = cp.array([1, 1, 1, 2, 2, 2], cp.float32)
        >>> from cuml.naive_bayes import GaussianNB
        >>> clf = GaussianNB()
        >>> clf.fit(X, Y)
        GaussianNB()
        >>> print(clf.predict(cp.array([[-0.8, -1]], cp.float32)))
        [1]
        >>> clf_pf = GaussianNB()
        >>> clf_pf.partial_fit(X, Y, cp.unique(Y))
        GaussianNB()
        >>> print(clf_pf.predict(cp.array([[-0.8, -1]], cp.float32)))
        [1]
    """

    def __init__(
        self,
        *,
        priors=None,
        var_smoothing=1e-9,
        output_type=None,
        handle=None,
        verbose=False,
    ):

        super(GaussianNB, self).__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.fit_called_ = False
        self.classes_ = None

    def fit(self, X, y, sample_weight=None) -> "GaussianNB":
        """
        Fit Gaussian Naive Bayes classifier according to X, y

        Parameters
        ----------

        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like shape (n_samples) Target values.
        sample_weight : array-like of shape (n_samples)
            Weights applied to individual samples (1. for unweighted).
            Currently sample weight is ignored.
        """
        return self._partial_fit(
            X,
            y,
            _classes=cp.unique(y),
            _refit=True,
            sample_weight=sample_weight,
        )

    @nvtx_annotate(
        message="naive_bayes.GaussianNB._partial_fit", domain="cuml_python"
    )
    def _partial_fit(
        self,
        X,
        y,
        _classes=None,
        _refit=False,
        sample_weight=None,
        convert_dtype=True,
    ) -> "GaussianNB":
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.internals.import_utils import (
                dummy_function_always_false as scipy_sparse_isspmatrix,
            )

        if getattr(self, "classes_") is None and _classes is None:
            raise ValueError(
                "classes must be passed on the first call " "to partial_fit."
            )

        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
        else:
            X = input_to_cupy_array(
                X, order="K", check_dtype=[cp.float32, cp.float64, cp.int32]
            ).array

        expected_y_dtype = (
            cp.int32 if X.dtype in [cp.float32, cp.int32] else cp.int64
        )
        y = input_to_cupy_array(
            y,
            convert_to_dtype=(expected_y_dtype if convert_dtype else False),
            check_dtype=expected_y_dtype,
        ).array

        if _classes is not None:
            _classes, *_ = input_to_cuml_array(
                _classes,
                order="K",
                convert_to_dtype=(
                    expected_y_dtype if convert_dtype else False
                ),
            )

        Y, label_classes = make_monotonic(y, classes=_classes, copy=True)
        if _refit:
            self.classes_ = None

        def var_sparse(X, axis=0):
            # Compute the variance on dense and sparse matrices
            return ((X - X.mean(axis=axis)) ** 2).mean(axis=axis)

        self.epsilon_ = self.var_smoothing * var_sparse(X).max()

        if not self.fit_called_:
            self.fit_called_ = True

            # Original labels are stored on the instance
            if _classes is not None:
                check_labels(Y, _classes.to_output("cupy"))
                self.classes_ = _classes
            else:
                self.classes_ = label_classes

            n_features = X.shape[1]
            n_classes = len(self.classes_)

            self.n_classes_ = n_classes
            self.n_features_ = n_features

            self.theta_ = cp.zeros((n_classes, n_features))
            self.sigma_ = cp.zeros((n_classes, n_features))

            self.class_count_ = cp.zeros(n_classes, dtype=X.dtype)

            if self.priors is not None:
                if len(self.priors) != n_classes:
                    raise ValueError(
                        "Number of priors must match number of" " classes."
                    )
                if not cp.isclose(self.priors.sum(), 1):
                    raise ValueError("The sum of the priors should be 1.")
                if (self.priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior, *_ = input_to_cupy_array(
                    self.priors, check_dtype=[cp.float32, cp.float64]
                )

        else:
            self.sigma_[:, :] -= self.epsilon_

        unique_y = cp.unique(y)
        unique_y_in_classes = cp.in1d(unique_y, cp.array(self.classes_))

        if not cp.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist "
                "in the initial classes %s"
                % (unique_y[~unique_y_in_classes], self.classes_)
            )

        self.theta_, self.sigma_ = self._update_mean_variance(X, Y)

        self.sigma_[:, :] += self.epsilon_

        if self.priors is None:
            self.class_prior = self.class_count_ / self.class_count_.sum()

        return self

    def partial_fit(
        self, X, y, classes=None, sample_weight=None
    ) -> "GaussianNB":
        """
        Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively on
        different chunks of a dataset so as to implement out-of-core or online
        learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible (as long
        as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------

        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features. A sparse matrix in COO
            format is preferred, other formats will go through a conversion
            to COO.
        y : array-like of shape (n_samples) Target values.
        classes : array-like of shape (n_classes)
                  List of all the classes that can possibly appear in the y
                  vector. Must be provided at the first call to partial_fit,
                  can be omitted in subsequent calls.
        sample_weight : array-like of shape (n_samples)
                        Weights applied to individual samples (1. for
                        unweighted). Currently sample weight is ignored.

        Returns
        -------
        self : object
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

    def _update_mean_variance(self, X, Y, sample_weight=None):

        if sample_weight is None:
            sample_weight = cp.zeros(0)

        labels_dtype = self.classes_.dtype

        mu = self.theta_
        var = self.sigma_

        early_return = self.class_count_.sum() == 0
        n_past = cp.expand_dims(self.class_count_, axis=1).copy()
        tpb = 32
        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if X.shape[0] == 0:
            return mu, var

        # Make sure Y is cp array not CumlArray
        Y = cp.asarray(Y)

        new_mu = cp.zeros(
            (self.n_classes_, self.n_features_), order="F", dtype=X.dtype
        )
        new_var = cp.zeros(
            (self.n_classes_, self.n_features_), order="F", dtype=X.dtype
        )
        class_counts = cp.zeros(self.n_classes_, order="F", dtype=X.dtype)
        if cupyx.scipy.sparse.isspmatrix(X):
            X = X.tocoo()

            count_features_coo = count_features_coo_kernel(
                X.dtype, labels_dtype
            )

            # Run once for averages
            count_features_coo(
                (math.ceil(X.nnz / tpb),),
                (tpb,),
                (
                    new_mu,
                    X.row,
                    X.col,
                    X.data,
                    X.nnz,
                    n_rows,
                    n_cols,
                    Y,
                    sample_weight,
                    sample_weight.shape[0] > 0,
                    self.n_classes_,
                    False,
                ),
            )

            # Run again for variance
            count_features_coo(
                (math.ceil(X.nnz / tpb),),
                (tpb,),
                (
                    new_var,
                    X.row,
                    X.col,
                    X.data,
                    X.nnz,
                    n_rows,
                    n_cols,
                    Y,
                    sample_weight,
                    sample_weight.shape[0] > 0,
                    self.n_classes_,
                    True,
                ),
            )
        else:

            count_features_dense = count_features_dense_kernel(
                X.dtype, labels_dtype
            )

            # Run once for averages
            count_features_dense(
                (math.ceil(n_rows / tpb), math.ceil(n_cols / tpb), 1),
                (tpb, tpb, 1),
                (
                    new_mu,
                    X,
                    n_rows,
                    n_cols,
                    Y,
                    sample_weight,
                    sample_weight.shape[0] > 0,
                    self.n_classes_,
                    False,
                    X.flags["C_CONTIGUOUS"],
                    False,
                ),
            )

            # Run again for variance
            count_features_dense(
                (math.ceil(n_rows / tpb), math.ceil(n_cols / tpb), 1),
                (tpb, tpb, 1),
                (
                    new_var,
                    X,
                    n_rows,
                    n_cols,
                    Y,
                    sample_weight,
                    sample_weight.shape[0] > 0,
                    self.n_classes_,
                    True,
                    X.flags["C_CONTIGUOUS"],
                    False,
                ),
            )

        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes(
            (math.ceil(n_rows / tpb),), (tpb,), (class_counts, n_rows, Y)
        )

        self.class_count_ += class_counts
        # Avoid any division by zero
        class_counts = cp.expand_dims(class_counts, axis=1)
        class_counts += cp.finfo(X.dtype).eps

        new_mu /= class_counts

        # Construct variance from sum squares
        new_var = (new_var / class_counts) - new_mu**2

        if early_return:
            return new_mu, new_var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight.shape[0] > 0:
            n_new = float(sample_weight.sum())
        else:
            n_new = class_counts

        n_total = n_past + n_new
        total_mu = (new_mu * n_new + mu * n_past) / n_total

        old_ssd = var * n_past
        new_ssd = n_new * new_var

        ssd_sum = old_ssd + new_ssd
        combined_feature_counts = n_new * n_past / n_total
        mean_adj = (mu - new_mu) ** 2

        total_ssd = ssd_sum + combined_feature_counts * mean_adj

        total_var = total_ssd / n_total
        return total_mu, total_var

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []

        for i in range(len(self.classes_)):
            jointi = cp.log(self.class_prior[i])

            n_ij = -0.5 * cp.sum(cp.log(2.0 * cp.pi * self.sigma_[i, :]))

            centered = (X - self.theta_[i, :]) ** 2
            zvals = centered / self.sigma_[i, :]
            summed = cp.sum(zvals, axis=1)

            n_ij = -(0.5 * summed) + n_ij
            joint_log_likelihood.append(jointi + n_ij)

        return cp.array(joint_log_likelihood).T

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ["priors", "var_smoothing"]


class _BaseDiscreteNB(_BaseNB):
    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        verbose=False,
        handle=None,
        output_type=None,
    ):
        super(_BaseDiscreteNB, self).__init__(
            verbose=verbose, handle=handle, output_type=output_type
        )
        if class_prior is not None:
            self.class_prior, *_ = input_to_cuml_array(class_prior)
        else:
            self.class_prior = None

        if alpha < 0:
            raise ValueError("Smoothing parameter alpha should be >= 0.")
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.fit_called_ = False
        self.n_classes_ = 0
        self.n_features_ = None

        # Needed until Base no longer assumed cumlHandle
        self.handle = None

    def _check_X_y(self, X, y):
        return X, y

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self.n_classes_:
                raise ValueError(
                    "Number of classes must match " "number of priors"
                )

            self.class_log_prior_ = cp.log(class_prior)

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)
            self.class_log_prior_ = log_class_count - cp.log(
                self.class_count_.sum()
            )
        else:
            self.class_log_prior_ = cp.full(
                self.n_classes_, -math.log(self.n_classes_)
            )

    def partial_fit(
        self, X, y, classes=None, sample_weight=None
    ) -> "_BaseDiscreteNB":
        """
        Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively on
        different chunks of a dataset so as to implement out-of-core or online
        learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible (as long
        as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------

        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features

        y : array-like of shape (n_samples) Target values.
        classes : array-like of shape (n_classes)
                  List of all the classes that can possibly appear in the y
                  vector. Must be provided at the first call to partial_fit,
                  can be omitted in subsequent calls.

        sample_weight : array-like of shape (n_samples)
                        Weights applied to individual samples (1. for
                        unweighted). Currently sample weight is ignored.

        Returns
        -------

        self : object
        """
        return self._partial_fit(
            X, y, sample_weight=sample_weight, _classes=classes
        )

    @nvtx_annotate(
        message="naive_bayes._BaseDiscreteNB._partial_fit",
        domain="cuml_python",
    )
    def _partial_fit(
        self, X, y, sample_weight=None, _classes=None, convert_dtype=True
    ) -> "_BaseDiscreteNB":
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.internals.import_utils import (
                dummy_function_always_false as scipy_sparse_isspmatrix,
            )

        # TODO: use SparseCumlArray
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
        else:
            X = input_to_cupy_array(
                X, order="K", check_dtype=[cp.float32, cp.float64, cp.int32]
            ).array

        expected_y_dtype = (
            cp.int32 if X.dtype in [cp.float32, cp.int32] else cp.int64
        )
        y = input_to_cupy_array(
            y,
            convert_to_dtype=(expected_y_dtype if convert_dtype else False),
            check_dtype=expected_y_dtype,
        ).array
        if _classes is not None:
            _classes, *_ = input_to_cuml_array(
                _classes,
                order="K",
                convert_to_dtype=(
                    expected_y_dtype if convert_dtype else False
                ),
            )
        Y, label_classes = make_monotonic(y, classes=_classes, copy=True)

        X, Y = self._check_X_y(X, Y)

        if not self.fit_called_:
            self.fit_called_ = True
            if _classes is not None:
                check_labels(Y, _classes.to_output("cupy"))
                self.classes_ = _classes
            else:
                self.classes_ = label_classes

            self.n_classes_ = self.classes_.shape[0]
            self.n_features_ = X.shape[1]
            self._init_counters(self.n_classes_, self.n_features_, X.dtype)
        else:
            check_labels(Y, self.classes_)

        if cupyx.scipy.sparse.isspmatrix(X):
            # X is assumed to be a COO here
            self._count_sparse(X.row, X.col, X.data, X.shape, Y, self.classes_)
        else:
            self._count(X, Y, self.classes_)

        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def fit(self, X, y, sample_weight=None) -> "_BaseDiscreteNB":
        """
        Fit Naive Bayes classifier according to X, y

        Parameters
        ----------

        X : {array-like, cupy sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like shape (n_samples) Target values.
        sample_weight : array-like of shape (n_samples)
            Weights applied to individual samples (1. for unweighted).
            Currently sample weight is ignored.
        """
        self.fit_called_ = False
        return self.partial_fit(X, y, sample_weight)

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(
            n_effective_classes, order="F", dtype=dtype
        )
        self.feature_count_ = cp.zeros(
            (n_effective_classes, n_features), order="F", dtype=dtype
        )

    def update_log_probs(self):
        """
        Updates the log probabilities. This enables lazy update for
        applications like distributed Naive Bayes, so that the model
        can be updated incrementally without incurring this cost each
        time.
        """
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

    def _count(self, X, Y, classes):
        """
        Sum feature counts & class prior counts and add to current model.
        Parameters
        ----------
        X : cupy.ndarray or cupyx.scipy.sparse matrix of size
                  (n_rows, n_features)
        Y : cupy.array of monotonic class labels
        """

        n_classes = classes.shape[0]
        sample_weight = cp.zeros(0)

        if X.ndim != 2:
            raise ValueError("Input samples should be a 2D array")

        if Y.dtype != classes.dtype:
            warnings.warn(
                "Y dtype does not match classes_ dtype. Y will be "
                "converted, which will increase memory consumption"
            )

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros(
            (n_classes, self.n_features_), order="F", dtype=X.dtype
        )

        class_c = cp.zeros(n_classes, order="F", dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        tpb = 32
        labels_dtype = classes.dtype

        count_features_dense = count_features_dense_kernel(
            X.dtype, labels_dtype
        )
        count_features_dense(
            (math.ceil(n_rows / tpb), math.ceil(n_cols / tpb), 1),
            (tpb, tpb, 1),
            (
                counts,
                X,
                n_rows,
                n_cols,
                Y,
                sample_weight,
                sample_weight.shape[0] > 0,
                n_classes,
                False,
                X.flags["C_CONTIGUOUS"],
                False,
            ),
        )

        tpb = 256
        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb),), (tpb,), (class_c, n_rows, Y))

        self.feature_count_ += counts
        self.class_count_ += class_c

    def _count_sparse(
        self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y, classes
    ):
        """
        Sum feature counts & class prior counts and add to current model.
        Parameters
        ----------
        x_coo_rows : cupy.ndarray of size (nnz)
        x_coo_cols : cupy.ndarray of size (nnz)
        x_coo_data : cupy.ndarray of size (nnz)
        Y : cupy.array of monotonic class labels
        """
        n_classes = classes.shape[0]

        if Y.dtype != classes.dtype:
            warnings.warn(
                "Y dtype does not match classes_ dtype. Y will be "
                "converted, which will increase memory consumption"
            )
        sample_weight = cp.zeros(0)

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros(
            (n_classes, self.n_features_), order="F", dtype=x_coo_data.dtype
        )

        class_c = cp.zeros(n_classes, order="F", dtype=x_coo_data.dtype)

        n_rows = x_shape[0]
        n_cols = x_shape[1]

        tpb = 256

        labels_dtype = classes.dtype

        count_features_coo = count_features_coo_kernel(
            x_coo_data.dtype, labels_dtype
        )
        count_features_coo(
            (math.ceil(x_coo_rows.shape[0] / tpb),),
            (tpb,),
            (
                counts,
                x_coo_rows,
                x_coo_cols,
                x_coo_data,
                x_coo_rows.shape[0],
                n_rows,
                n_cols,
                Y,
                sample_weight,
                sample_weight.shape[0] > 0,
                n_classes,
                False,
            ),
        )

        count_classes = count_classes_kernel(x_coo_data.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb),), (tpb,), (class_c, n_rows, Y))

        self.feature_count_ = self.feature_count_ + counts
        self.class_count_ = self.class_count_ + class_c

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "alpha",
            "fit_prior",
            "class_prior",
        ]


class MultinomialNB(_BaseDiscreteNB):

    # TODO: Make this extend cuml.Base:
    # https://github.com/rapidsai/cuml/issues/1834

    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification
    with discrete features (e.g., word counts for text classification).

    The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------

    alpha : float (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no
        smoothing).
    fit_prior : boolean (default=True)
        Whether to learn class prior probabilities or no. If false, a
        uniform prior will be used.
    class_prior : array-like, size (n_classes) (default=None)
        Prior probabilities of the classes. If specified, the priors are
        not adjusted according to the data.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------

    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    .. code-block:: python

        >>> import cupy as cp
        >>> import cupyx
        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> from cuml.naive_bayes import MultinomialNB

        >>> # Load corpus
        >>> twenty_train = fetch_20newsgroups(subset='train', shuffle=True,
        ...                                   random_state=42)

        >>> # Turn documents into term frequency vectors

        >>> count_vect = CountVectorizer()
        >>> features = count_vect.fit_transform(twenty_train.data)

        >>> # Put feature vectors and labels on the GPU

        >>> X = cupyx.scipy.sparse.csr_matrix(features.tocsr(),
        ...                                   dtype=cp.float32)
        >>> y = cp.asarray(twenty_train.target, dtype=cp.int32)

        >>> # Train model

        >>> model = MultinomialNB()
        >>> model.fit(X, y)
        MultinomialNB()

        >>> # Compute accuracy on training set

        >>> model.score(X, y)
        0.9245...

    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        output_type=None,
        handle=None,
        verbose=False,
    ):
        super(MultinomialNB, self).__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
        )

    def _update_feature_log_prob(self, alpha):
        """
        Apply add-lambda smoothing to raw counts and recompute
        log probabilities

        Parameters
        ----------

        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = cp.log(smoothed_fc) - cp.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _joint_log_likelihood(self, X):
        """
        Calculate the posterior log probability of the samples X

        Parameters
        ----------

        X : array-like of size (n_samples, n_features)
        """
        ret = X.dot(self.feature_log_prob_.T)
        ret += self.class_log_prior_
        return ret


class BernoulliNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for multivariate Bernoulli models.
    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Parameters
    ----------

    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> rng = cp.random.RandomState(1)
        >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
        >>> Y = cp.array([1, 2, 3, 4, 4, 5])
        >>> from cuml.naive_bayes import BernoulliNB
        >>> clf = BernoulliNB()
        >>> clf.fit(X, Y)
        BernoulliNB()
        >>> print(clf.predict(X[2:3]))
        [3]

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.
    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        binarize=0.0,
        fit_prior=True,
        class_prior=None,
        output_type=None,
        handle=None,
        verbose=False,
    ):
        super(BernoulliNB, self).__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
        )
        self.binarize = binarize

    def _check_X(self, X):
        X = super()._check_X(X)
        if self.binarize is not None:
            if cupyx.scipy.sparse.isspmatrix(X):
                X.data = binarize(X.data, threshold=self.binarize)
            else:
                X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if self.binarize is not None:
            if cupyx.scipy.sparse.isspmatrix(X):
                X.data = binarize(X.data, threshold=self.binarize)
            else:
                X = binarize(X, threshold=self.binarize)
        return X, y

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        neg_prob = cp.log(1 - cp.exp(self.feature_log_prob_))

        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = X.dot((self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll

    def _update_feature_log_prob(self, alpha):
        """
        Apply add-lambda smoothing to raw counts and recompute
        log probabilities

        Parameters
        ----------

        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2
        self.feature_log_prob_ = cp.log(smoothed_fc) - cp.log(
            smoothed_cc.reshape(-1, 1)
        )

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ["binarize"]


class ComplementNB(_BaseDiscreteNB):
    """
    The Complement Naive Bayes classifier described in Rennie et al. (2003).
    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Parameters
    ----------

    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    norm : bool, default=False
        Whether or not a second normalization of the weights is performed.
        The default behavior mirrors the implementation found in Mahout and
        Weka, which do not follow the full algorithm described in Table 9 of
        the paper.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> rng = cp.random.RandomState(1)
        >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
        >>> Y = cp.array([1, 2, 3, 4, 4, 5])
        >>> from cuml.naive_bayes import ComplementNB
        >>> clf = ComplementNB()
        >>> clf.fit(X, Y)
        ComplementNB()
        >>> print(clf.predict(X[2:3]))
        [3]

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        norm=False,
        output_type=None,
        handle=None,
        verbose=False,
    ):
        super(ComplementNB, self).__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
        )
        self.norm = norm

    def _check_X(self, X):
        X = super()._check_X(X)
        if cupyx.scipy.sparse.isspmatrix(X):
            X_min = X.data.min()
        else:
            X_min = X.min()
        if X_min < 0:
            raise ValueError("Negative values in data passed to ComplementNB")
        return X

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if cupyx.scipy.sparse.isspmatrix(X):
            X_min = X.data.min()
        else:
            X_min = X.min()
        if X_min < 0:
            raise ValueError("Negative values in data passed to ComplementNB")
        return X, y

    def _count(self, X, Y, classes):
        super()._count(X, Y, classes)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _count_sparse(
        self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y, classes
    ):
        super()._count_sparse(
            x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y, classes
        )
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _joint_log_likelihood(self, X):
        """Calculate the class scores for the samples in X."""
        jll = X.dot(self.feature_log_prob_.T)
        if len(self.class_count_) == 1:
            jll += self.class_log_prior_
        return jll

    def _update_feature_log_prob(self, alpha):
        """
        Apply smoothing to raw counts and compute the weights.

        Parameters
        ----------

        alpha : float amount of smoothing to apply (0. means no smoothing)
        """
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = cp.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ["norm"]


class CategoricalNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for categorical features
    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    category_count_ : ndarray of shape (n_features, n_classes, n_categories)
        With n_categories being the highest category of all the features.
        This array provides the number of samples encountered for each feature,
        class and category of the specific feature.
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.
    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_log_prob_ : ndarray of shape (n_features, n_classes, n_categories)
        With n_categories being the highest category of all the features.
        Each array of shape (n_classes, n_categories) provides the empirical
        log probability of categories given the respective feature
        and class, ``P(x_i|y)``.
        This attribute is not available when the model has been trained with
        sparse data.
    n_features_ : int
        Number of features of each sample.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> rng = cp.random.RandomState(1)
        >>> X = rng.randint(5, size=(6, 100), dtype=cp.int32)
        >>> y = cp.array([1, 2, 3, 4, 5, 6])
        >>> from cuml.naive_bayes import CategoricalNB
        >>> clf = CategoricalNB()
        >>> clf.fit(X, y)
        CategoricalNB()
        >>> print(clf.predict(X[2:3]))
        [3]
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
        output_type=None,
        handle=None,
        verbose=False,
    ):
        super(CategoricalNB, self).__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
        )

    def _check_X_y(self, X, y):
        if cupyx.scipy.sparse.isspmatrix(X):
            warnings.warn(
                "X dtype is not int32. X will be "
                "converted, which will increase memory consumption"
            )
            X.data = X.data.astype(cp.int32)
            x_min = X.data.min()
        else:
            if X.dtype not in [cp.int32]:
                warnings.warn(
                    "X dtype is not int32. X will be "
                    "converted, which will increase memory "
                    "consumption"
                )
                X = input_to_cupy_array(
                    X, order="K", convert_to_dtype=cp.int32
                ).array
            x_min = X.min()
        if x_min < 0:
            raise ValueError("Negative values in data passed to CategoricalNB")
        return X, y

    def _check_X(self, X):
        if cupyx.scipy.sparse.isspmatrix(X):
            warnings.warn(
                "X dtype is not int32. X will be "
                "converted, which will increase memory consumption"
            )
            X.data = X.data.astype(cp.int32)
            x_min = X.data.min()
        else:
            if X.dtype not in [cp.int32]:
                warnings.warn(
                    "X dtype is not int32. X will be "
                    "converted, which will increase memory "
                    "consumption"
                )
                X = input_to_cupy_array(
                    X, order="K", convert_to_dtype=cp.int32
                ).array
            x_min = X.min()
        if x_min < 0:
            raise ValueError("Negative values in data passed to CategoricalNB")
        return X

    def fit(self, X, y, sample_weight=None) -> "CategoricalNB":
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples), default=None
            Weights applied to individual samples (1. for unweighted).
            Currently sample weight is ignored.

        Returns
        -------
        self : object
        """
        return super().fit(X, y, sample_weight=sample_weight)

    def partial_fit(
        self, X, y, classes=None, sample_weight=None
    ) -> "CategoricalNB":
        """Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.
        y : array-like of shape (n_samples)
            Target values.
        classes : array-like of shape (n_classes), default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        sample_weight : array-like of shape (n_samples), default=None
            Weights applied to individual samples (1. for unweighted).
            Currently sample weight is ignored.

        Returns
        -------
        self : object
        """
        return super().partial_fit(X, y, classes, sample_weight=sample_weight)

    def _count_sparse(
        self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y, classes
    ):
        """
        Sum feature counts & class prior counts and add to current model.
        Parameters
        ----------
        x_coo_rows : cupy.ndarray of size (nnz)
        x_coo_cols : cupy.ndarray of size (nnz)
        x_coo_data : cupy.ndarray of size (nnz)
        Y : cupy.array of monotonic class labels
        """
        n_classes = classes.shape[0]
        n_rows = x_shape[0]
        n_cols = x_shape[1]
        x_coo_nnz = x_coo_rows.shape[0]
        labels_dtype = classes.dtype
        tpb = 256

        if Y.dtype != classes.dtype:
            warnings.warn(
                "Y dtype does not match classes_ dtype. Y will be "
                "converted, which will increase memory consumption"
            )

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        class_c = cp.zeros(n_classes, dtype=self.class_count_.dtype)
        count_classes = count_classes_kernel(
            self.class_count_.dtype, labels_dtype
        )
        count_classes((math.ceil(n_rows / tpb),), (tpb,), (class_c, n_rows, Y))

        highest_feature = int(x_coo_data.max()) + 1
        feature_diff = highest_feature - self.category_count_.shape[1]
        # In case of a partial fit, pad the array to have the highest feature
        if not cupyx.scipy.sparse.issparse(self.category_count_):
            self.category_count_ = cupyx.scipy.sparse.coo_matrix(
                (self.n_features_ * n_classes, highest_feature)
            )
        elif feature_diff > 0:
            self.category_count_ = cupyx.scipy.sparse.coo_matrix(
                self.category_count_,
                shape=(self.n_features_ * n_classes, highest_feature),
            )
        highest_feature = self.category_count_.shape[1]

        count_features_coo = cp.ElementwiseKernel(
            "int32 row, int32 col, int32 val, int32 nnz, int32 n_classes, \
             int32 n_cols, raw T labels",
            "int32 out_row, int32 out_col",
            """
            T label = labels[row];
            out_row = col + n_cols * label;
            out_col = val;
            """,
            "count_features_categorical_coo_kernel",
        )
        counts_rows, counts_cols = count_features_coo(
            x_coo_rows, x_coo_cols, x_coo_data, x_coo_nnz, n_classes, n_cols, Y
        )
        # Create the sparse category count matrix from the result of
        # the raw kernel
        counts = cupyx.scipy.sparse.coo_matrix(
            (cp.ones(x_coo_nnz), (counts_rows, counts_cols)),
            shape=(self.n_features_ * n_classes, highest_feature),
        ).tocsr()

        # Adjust with the missing (zeros) data of the sparse matrix
        for i in range(n_classes):
            counts[i * n_cols : (i + 1) * n_cols, 0] = (Y == i).sum() - counts[
                i * n_cols : (i + 1) * n_cols
            ].sum(1)
        self.category_count_ = (self.category_count_ + counts).tocoo()
        self.class_count_ = self.class_count_ + class_c

    def _count(self, X, Y, classes):
        Y = cp.asarray(Y)
        tpb = 32
        n_rows = X.shape[0]
        n_cols = X.shape[1]
        n_classes = classes.shape[0]
        labels_dtype = classes.dtype

        sample_weight = cp.zeros(0, dtype=X.dtype)
        highest_feature = int(X.max()) + 1
        feature_diff = highest_feature - self.category_count_.shape[2]
        # In case of a partial fit, pad the array to have the highest feature
        if feature_diff > 0:
            self.category_count_ = cp.pad(
                self.category_count_,
                [(0, 0), (0, 0), (0, feature_diff)],
                "constant",
            )
        highest_feature = self.category_count_.shape[2]
        counts = cp.zeros(
            (self.n_features_, n_classes, highest_feature),
            order="F",
            dtype=X.dtype,
        )

        count_features = count_features_dense_kernel(X.dtype, Y.dtype)
        count_features(
            (math.ceil(n_rows / tpb), math.ceil(n_cols / tpb), 1),
            (tpb, tpb, 1),
            (
                counts,
                X,
                n_rows,
                n_cols,
                Y,
                sample_weight,
                sample_weight.shape[0] > 0,
                self.n_classes_,
                False,
                X.flags["C_CONTIGUOUS"],
                True,
            ),
        )
        self.category_count_ += counts

        class_c = cp.zeros(n_classes, order="F", dtype=self.class_count_.dtype)
        count_classes = count_classes_kernel(class_c.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb),), (tpb,), (class_c, n_rows, Y))
        self.class_count_ += class_c

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(
            n_effective_classes, order="F", dtype=cp.float64
        )
        self.category_count_ = cp.zeros(
            (n_features, n_effective_classes, 0), order="F", dtype=dtype
        )

    def _update_feature_log_prob(self, alpha):
        highest_feature = cp.zeros(self.n_features_, dtype=cp.float64)
        if cupyx.scipy.sparse.issparse(self.category_count_):
            # For sparse data we avoid the creation of the dense matrix
            # feature_log_prob_. This can be created on the fly during
            # the prediction without using as much memory.
            features = self.category_count_.row % self.n_features_
            cupyx.scatter_max(
                highest_feature, features, self.category_count_.col
            )
            highest_feature = (highest_feature + 1) * alpha

            smoothed_class_count = self.category_count_.sum(axis=1)
            smoothed_class_count = smoothed_class_count.reshape(
                (self.n_classes_, self.n_features_)
            ).T
            smoothed_class_count += highest_feature[:, cp.newaxis]
            smoothed_cat_count = cupyx.scipy.sparse.coo_matrix(
                self.category_count_
            )
            smoothed_cat_count.data = cp.log(smoothed_cat_count.data + alpha)
            self.smoothed_cat_count = smoothed_cat_count.tocsr()
            self.smoothed_class_count = cp.log(smoothed_class_count)
        else:
            indices = self.category_count_.nonzero()
            cupyx.scatter_max(highest_feature, indices[0], indices[2])
            highest_feature = (highest_feature + 1) * alpha

            smoothed_class_count = (
                self.category_count_.sum(axis=2)
                + highest_feature[:, cp.newaxis]
            )
            smoothed_cat_count = self.category_count_ + alpha
            self.feature_log_prob_ = cp.log(smoothed_cat_count) - cp.log(
                smoothed_class_count[:, :, cp.newaxis]
            )

    def _joint_log_likelihood(self, X):
        if not X.shape[1] == self.n_features_:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (self.n_features_, X.shape[1])
            )
        n_rows = X.shape[0]
        if cupyx.scipy.sparse.isspmatrix(X):
            # For sparse data we assume that most categories will be zeros,
            # so we first compute the jll for categories 0
            features_zeros = self.smoothed_cat_count[:, 0].todense()
            features_zeros = features_zeros.reshape(
                self.n_classes_, self.n_features_
            ).T
            if self.alpha != 1.0:
                features_zeros[cp.where(features_zeros == 0)] += cp.log(
                    self.alpha
                )
            features_zeros -= self.smoothed_class_count
            features_zeros = features_zeros.sum(0)
            jll = cp.repeat(features_zeros[cp.newaxis, :], n_rows, axis=0)

            X = X.tocoo()
            col_indices = X.col

            # Adjust with the non-zeros data by adding jll_data (non-zeros)
            # and subtracting jll_zeros which are the zeros
            # that were first computed
            for i in range(self.n_classes_):
                jll_data = self.smoothed_cat_count[
                    col_indices + i * self.n_features_, X.data
                ].ravel()
                jll_zeros = self.smoothed_cat_count[
                    col_indices + i * self.n_features_, 0
                ].todense()[:, 0]
                if self.alpha != 1.0:
                    jll_data[cp.where(jll_data == 0)] += cp.log(self.alpha)
                    jll_zeros[cp.where(jll_zeros == 0)] += cp.log(self.alpha)
                jll_data -= jll_zeros
                cupyx.scatter_add(jll[:, i], X.row, jll_data)

        else:
            col_indices = cp.indices(X.shape)[1].flatten()
            jll = self.feature_log_prob_[col_indices, :, X.ravel()]
            jll = jll.reshape((n_rows, self.n_features_, self.n_classes_))
            jll = jll.sum(1)
        jll += self.class_log_prior_
        return jll
