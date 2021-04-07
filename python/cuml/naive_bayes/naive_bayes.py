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
import math
import warnings

import cupy as cp
import cupy.prof
import cupyx
from cuml.common import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common.import_utils import has_scipy
from cuml.common.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.prims.label import check_labels, invert_labels, make_monotonic


def count_features_coo_kernel(float_dtype, int_dtype):
    """
    A simple reduction kernel that takes in a sparse (COO) array
    of features and computes the sum (or sum squared) for each class
    label
    """

    kernel_str = r'''({0} *out,
                    int *rows, int *cols,
                    {0} *vals, int nnz,
                    int n_rows, int n_cols,
                    {1} *labels,
                    int n_classes,
                    bool square) {

      int i = blockIdx.x * blockDim.x + threadIdx.x;

      if(i >= nnz) return;

      int row = rows[i];
      int col = cols[i];
      {0} val = vals[i];
      if(square) val *= val;
      {1} label = labels[row];
      atomicAdd(out + ((col * n_classes) + label), val);
    }'''

    return cuda_kernel_factory(kernel_str, (float_dtype, int_dtype),
                               "count_features_coo")


def count_classes_kernel(float_dtype, int_dtype):
    kernel_str = r'''
    ({0} *out, int n_rows, {1} *labels) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if(row >= n_rows) return;
      {1} label = labels[row];
      atomicAdd(out + label, 1);
    }'''

    return cuda_kernel_factory(kernel_str, (float_dtype, int_dtype),
                               "count_classes")


def count_features_dense_kernel(float_dtype, int_dtype):

    kernel_str = r'''
    ({0} *out,
     {0} *in,
     int n_rows,
     int n_cols,
     {1} *labels,
     int n_classes,
     bool square,
     bool rowMajor) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;

      if(row >= n_rows || col >= n_cols) return;

      {0} val = !rowMajor ?
            in[col * n_rows + row] : in[row * n_cols + col];

      if(val == 0.0) return;

      if(square) val *= val;
      {1} label = labels[row];

      atomicAdd(out + ((col * n_classes) + label), val);
    }'''

    return cuda_kernel_factory(kernel_str, (float_dtype, int_dtype),
                               "count_features_dense")


def _convert_x_sparse(X):
    X = X.tocoo()

    if X.dtype not in [cp.float32, cp.float64]:
        raise ValueError("Only floating-point dtypes (float32 or "
                         "float64) are supported for sparse inputs.")

    rows = cp.asarray(X.row, dtype=X.row.dtype)
    cols = cp.asarray(X.col, dtype=X.col.dtype)
    data = cp.asarray(X.data, dtype=X.data.dtype)
    return cupyx.scipy.sparse.coo_matrix((data, (rows, cols)),
                                         shape=X.shape)


class MultinomialNB(Base, ClassifierMixin):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification
    with discrete features (e.g., word counts for text classification).

    The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.

    Notes
    -----
    While cuML only provides the multinomial version currently, the other
    variants are planned to be included soon. Refer to the corresponding Github
    `issue <https://github.com/rapidsai/cuml/issues/1666>`_ for updates.

    Parameters
    ----------

    alpha : float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    fit_prior : boolean
        Whether to learn class prior probabilities or no. If false, a uniform
        prior will be used.
    class_prior : array-like, size (n_classes)
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Examples
    --------

    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    .. code-block:: python

        import cupy as cp
        import cupyx

        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer

        from cuml.naive_bayes import MultinomialNB

        # Load corpus

        twenty_train = fetch_20newsgroups(subset='train',
                                shuffle=True, random_state=42)

        # Turn documents into term frequency vectors

        count_vect = CountVectorizer()
        features = count_vect.fit_transform(twenty_train.data)

        # Put feature vectors and labels on the GPU

        X = cupyx.scipy.sparse.csr_matrix(features.tocsr(), dtype=cp.float32)
        y = cp.asarray(twenty_train.target, dtype=cp.int32)

        # Train model

        model = MultinomialNB()
        model.fit(X, y)

        # Compute accuracy on training set

        model.score(X, y)

    Output:

    .. code-block:: python

        0.9244298934936523

    """

    classes_ = CumlArrayDescriptor()
    class_count_ = CumlArrayDescriptor()
    feature_count_ = CumlArrayDescriptor()
    class_log_prior_ = CumlArrayDescriptor()
    feature_log_prob_ = CumlArrayDescriptor()

    def __init__(self, *,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None,
                 output_type=None,
                 handle=None,
                 verbose=False):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.alpha = alpha
        self.fit_prior = fit_prior

        if class_prior is not None:
            self._class_prior, *_ = input_to_cuml_array(class_prior)
        else:
            self._class_prior_ = None

        self.fit_called_ = False
        self._n_classes_ = 0
        self._n_features_ = None

        # Needed until Base no longer assumed cumlHandle
        self.handle = None

    @generate_docstring(X='dense_sparse')
    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    def fit(self, X, y,
            sample_weight=None, convert_dtype=True) -> "MultinomialNB":
        """
        Fit Naive Bayes classifier according to X, y
        """
        return self.partial_fit(X, y, sample_weight,
                                convert_dtype=convert_dtype)

    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    def _partial_fit(self,
                     X,
                     y,
                     sample_weight=None,
                     _classes=None,
                     convert_dtype=True) -> "MultinomialNB":

        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
            # TODO: Expanded this since sparse kernel doesn't
            # actually require the scipy sparse container format.
        else:
            X = input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64,
                                                 cp.int32]).array

        expected_y_dtype = cp.int32 if X.dtype in [cp.float32,
                                                   cp.int32] else cp.int64
        y = input_to_cupy_array(y,
                                convert_to_dtype=(expected_y_dtype
                                                  if convert_dtype
                                                  else False),
                                check_dtype=expected_y_dtype).array

        Y, label_classes = make_monotonic(y, copy=True)

        if not self.fit_called_:
            self.fit_called_ = True
            if _classes is not None:
                _classes, *_ = input_to_cuml_array(_classes,
                                                   order='K',
                                                   convert_to_dtype=(
                                                       expected_y_dtype
                                                       if convert_dtype
                                                       else False))
                check_labels(Y, _classes)
                self.classes_ = _classes
            else:
                self.classes_ = label_classes

            self._n_classes_ = self.classes_.shape[0]
            self._n_features_ = X.shape[1]
            self._init_counters(self._n_classes_, self._n_features_, X.dtype)
        else:
            check_labels(Y, self.classes_)

        if cp.sparse.isspmatrix(X):
            self._count_sparse(X.row, X.col, X.data, X.shape, Y)
        else:
            self._count(X, Y)

        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self._class_prior_)

        return self

    def update_log_probs(self):
        """
        Updates the log probabilities. This enables lazy update for
        applications like distributed Naive Bayes, so that the model
        can be updated incrementally without incurring this cost each
        time.
        """
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self._class_prior_)

    def partial_fit(self,
                    X,
                    y,
                    classes=None,
                    sample_weight=None,
                    convert_dtype=True) -> "MultinomialNB":
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

        y : array-like of int32 or int64, shape (n_samples)
            Target values.

        classes : array-like of shape (n_classes)
            List of all the classes that can possibly appear in the y
            vector. Must be provided at the first call to partial_fit,
            can be omitted in subsequent calls.

        sample_weight : array-like of shape (n_samples)
            Weights applied to individual samples (1. for
            unweighted). Currently sample weight is ignored

        convert_dtype : bool
            If True, convert y to the appropriate dtype (int)

        Returns
        -------

        self : object
        """
        return self._partial_fit(X,
                                 y,
                                 sample_weight=sample_weight,
                                 _classes=classes)

    @generate_docstring(X='dense_sparse',
                        return_values={
                            'name': 'y_hat',
                            'type': 'dense',
                            'description': 'Predicted values',
                            'shape': '(n_rows, 1)'
                        })
    @cp.prof.TimeRangeDecorator(message="predict()", color_id=1)
    def predict(self, X) -> CumlArray:
        """
        Perform classification on an array of test vectors X.

        """
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
        else:
            X = input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64,
                                                 cp.int32]).array

        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1).astype(self.classes_.dtype)

        y_hat = invert_labels(indices, classes=self.classes_)
        return y_hat

    @generate_docstring(
        X='dense_sparse',
        return_values={
            'name': 'C',
            'type': 'dense',
            'description': (
                'Returns the log-probability of the samples for each class in '
                'the model. The columns correspond to the classes in sorted '
                'order, as they appear in the attribute `classes_`.'),
            'shape': '(n_rows, 1)'
        })
    def predict_log_proba(self, X) -> CumlArray:
        """
        Return log-probability estimates for the test vector X.

        """
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cupyx.scipy.sparse.isspmatrix(X):
            X = _convert_x_sparse(X)
        else:
            X = input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32,
                                                 cp.float64,
                                                 cp.int32]).array

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
        return result

    @generate_docstring(
        X='dense_sparse',
        return_values={
            'name': 'C',
            'type': 'dense',
            'description': (
                'Returns the probability of the samples for each class in the '
                'model. The columns correspond to the classes in sorted order,'
                ' as they appear in the attribute `classes_`.'),
            'shape': '(n_rows, 1)'
        })
    def predict_proba(self, X) -> CumlArray:
        """
        Return probability estimates for the test vector X.

        """
        result = cp.exp(self.predict_log_proba(X))
        return result

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(n_effective_classes,
                                     order="F",
                                     dtype=dtype)
        self.feature_count_ = cp.zeros((n_effective_classes, n_features),
                                       order="F",
                                       dtype=dtype)

    def _count(self, X, Y):
        """
        Sum feature counts & class prior counts and add to current model.

        Parameters
        ----------
        X : cupy.ndarray or cupyx.scipy.sparse matrix of size
            (n_rows, n_features)
        Y : cupy.array of monotonic class labels
        """

        if X.ndim != 2:
            raise ValueError("Input samples should be a 2D array")

        if Y.dtype != self.classes_.dtype:
            warnings.warn("Y dtype does not match classes_ dtype. Y will be "
                          "converted, which will increase memory consumption")

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros((self._n_classes_, self._n_features_),
                          order="F",
                          dtype=X.dtype)

        class_c = cp.zeros(self._n_classes_, order="F", dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        tpb = 32
        labels_dtype = self.classes_.dtype

        count_features_dense = count_features_dense_kernel(
            X.dtype, labels_dtype)
        count_features_dense(
            (math.ceil(n_rows / tpb), math.ceil(n_cols / tpb), 1),
            (tpb, tpb, 1),
            (counts,
             X,
             n_rows,
             n_cols,
             Y,
             self._n_classes_,
             False,
             X.flags["C_CONTIGUOUS"]))

        tpb = 256
        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb), ), (tpb, ),
                      (class_c, n_rows, Y))

        self.feature_count_ = self.feature_count_ + counts
        self.class_count_ = self.class_count_ + class_c

    def _count_sparse(self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y):
        """
        Sum feature counts & class prior counts and add to current model.

        Parameters
        ----------
        x_coo_rows : cupy.ndarray of size (nnz)
        x_coo_cols : cupy.ndarray of size (nnz)
        x_coo_data : cupy.ndarray of size (nnz)
        Y : cupy.array of monotonic class labels
        """

        if Y.dtype != self.classes_.dtype:
            warnings.warn("Y dtype does not match classes_ dtype. Y will be "
                          "converted, which will increase memory consumption")

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros((self._n_classes_, self._n_features_),
                          order="F",
                          dtype=x_coo_data.dtype)

        class_c = cp.zeros(self._n_classes_, order="F", dtype=x_coo_data.dtype)

        n_rows = x_shape[0]
        n_cols = x_shape[1]

        tpb = 256

        labels_dtype = self.classes_.dtype

        count_features_coo = count_features_coo_kernel(
            x_coo_data.dtype, labels_dtype)
        count_features_coo((math.ceil(x_coo_rows.shape[0] / tpb), ), (tpb, ),
                           (counts,
                            x_coo_rows,
                            x_coo_cols,
                            x_coo_data,
                            x_coo_rows.shape[0],
                            n_rows,
                            n_cols,
                            Y,
                            self._n_classes_,
                            False))

        count_classes = count_classes_kernel(x_coo_data.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb), ), (tpb, ),
                      (class_c, n_rows, Y))

        self.feature_count_ = self.feature_count_ + counts
        self.class_count_ = self.class_count_ + class_c

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self._n_classes_:
                raise ValueError("Number of classes must match "
                                 "number of priors")

            self.class_log_prior_ = cp.log(class_prior)

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)

            self.class_log_prior_ = \
                log_class_count - cp.log(
                    self.class_count_.sum())
        else:
            self.class_log_prior_ = cp.full(self._n_classes_,
                                            -1 * math.log(self._n_classes_))

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
            smoothed_cc.reshape(-1, 1))

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

    def get_param_names(self):
        return super().get_param_names() + \
            [
                "alpha",
                "fit_prior",
                "class_prior",
            ]
