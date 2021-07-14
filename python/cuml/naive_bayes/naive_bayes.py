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
import nvtx

import cupy as cp
import cupyx
from cuml.common import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common.import_utils import has_scipy
from cuml.prims.label import make_monotonic
from cuml.prims.label import check_labels
from cuml.prims.label import invert_labels
from cuml.prims.array import binarize

from cuml.common.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.common.kernel_utils import cuda_kernel_factory


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
                    {0} *weights,
                    bool has_weights,
                    int n_classes,
                    bool square) {

      int i = blockIdx.x * blockDim.x + threadIdx.x;

      if(i >= nnz) return;

      int row = rows[i];
      int col = cols[i];
      {0} val = vals[i];

      if(has_weights)
        val *= weights[i];

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
      atomicAdd(out + label, ({0})1);
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
     {0} *weights,
     bool has_weights,
     int n_classes,
     bool square,
     bool rowMajor) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;

      if(row >= n_rows || col >= n_cols) return;

      {0} val = !rowMajor ?
            in[col * n_rows + row] : in[row * n_cols + col];

      if(has_weights)
        val *= weights[row];

      if(val == 0.0) return;

      if(square) val *= val;
      {1} label = labels[row];

      {1} idx = rowMajor ? col : row;
      atomicAdd(out + ((idx * n_classes) + label), val);
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


class _BaseNB(Base, ClassifierMixin):

    classes_ = CumlArrayDescriptor()
    class_count_ = CumlArrayDescriptor()
    feature_count_ = CumlArrayDescriptor()
    class_log_prior_ = CumlArrayDescriptor()
    feature_log_prob_ = CumlArrayDescriptor()

    def __init__(self, *, verbose=False, handle=None, output_type=None):
        super(_BaseNB, self).__init__(verbose=verbose,
                                      handle=handle,
                                      output_type=output_type)

    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks."""
        return X

    @generate_docstring(X='dense_sparse',
                        return_values={
                            'name': 'y_hat',
                            'type': 'dense',
                            'description': 'Predicted values',
                            'shape': '(n_rows, 1)'
                        })
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

        X = self._check_X(X)
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


class _BaseDiscreteNB(_BaseNB):

    def _check_X_y(self, X, y):
        return X, y

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self.n_classes_:
                raise ValueError("Number of classes must match "
                                 "number of priors")

            self.class_log_prior_ = cp.log(class_prior)

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)
            self.class_log_prior_ = log_class_count - \
                cp.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = cp.full(self.n_classes_,
                                            -math.log(self.n_classes_))

    def partial_fit(self, X, y, classes=None,
                    sample_weight=None) -> "_BaseDiscreteNB":
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
                        unweighted). Currently sample weight is ignored

        Returns
        -------

        self : object
        """
        return self._partial_fit(X, y, sample_weight=sample_weight,
                                 _classes=classes)

    @nvtx.annotate(message="naive_bayes._BaseDiscreteNB._partial_fit",
                   domain="cuml_python")
    def _partial_fit(self, X, y, sample_weight=None,
                     _classes=None, convert_dtype=True) -> "_BaseDiscreteNB":
        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cp.sparse.isspmatrix(X):
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

        X, Y = self._check_X_y(X, Y)

        if not self.fit_called_:
            self.fit_called_ = True
            if _classes is not None:
                _classes, *_ = input_to_cuml_array(_classes, order='K',
                                                   convert_to_dtype=(
                                                       expected_y_dtype
                                                       if convert_dtype
                                                       else False))
                check_labels(Y, _classes.to_output('cupy'))
                self.classes_ = _classes
            else:
                self.classes_ = label_classes

            self.n_classes_ = self.classes_.shape[0]
            self.n_features_ = X.shape[1]
            self._init_counters(self.n_classes_, self.n_features_,
                                X.dtype)
        else:
            check_labels(Y, self.classes_)

        if cp.sparse.isspmatrix(X):
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
            Weights applied to individial samples (1. for unweighted).
        """
        return self.partial_fit(X, y, sample_weight)

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self.class_count_ = cp.zeros(n_effective_classes,
                                     order="F",
                                     dtype=dtype)
        self.feature_count_ = cp.zeros((n_effective_classes, n_features),
                                       order="F",
                                       dtype=dtype)

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
        X : cupy.ndarray or cupy.sparse matrix of size
                  (n_rows, n_features)
        Y : cupy.array of monotonic class labels
        """

        n_classes = classes.shape[0]
        sample_weight = cp.zeros(0)

        if X.ndim != 2:
            raise ValueError("Input samples should be a 2D array")

        if Y.dtype != classes.dtype:
            warnings.warn("Y dtype does not match classes_ dtype. Y will be "
                          "converted, which will increase memory consumption")

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros((n_classes, self.n_features_), order="F",
                          dtype=X.dtype)

        class_c = cp.zeros(n_classes, order="F", dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        tpb = 32
        labels_dtype = classes.dtype

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
             sample_weight,
             sample_weight.shape[0] > 0,
             n_classes,
             False,
             X.flags["C_CONTIGUOUS"]))

        tpb = 256
        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb),), (tpb,),
                      (class_c, n_rows, Y))

        self.feature_count_ += counts
        self.class_count_ += class_c

    def _count_sparse(self, x_coo_rows, x_coo_cols, x_coo_data, x_shape, Y,
                      classes):
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
            warnings.warn("Y dtype does not match classes_ dtype. Y will be "
                          "converted, which will increase memory consumption")
        sample_weight = cp.zeros(0)

        # Make sure Y is a cupy array, not CumlArray
        Y = cp.asarray(Y)

        counts = cp.zeros((n_classes, self.n_features_),
                          order="F",
                          dtype=x_coo_data.dtype)

        class_c = cp.zeros(n_classes, order="F", dtype=x_coo_data.dtype)

        n_rows = x_shape[0]
        n_cols = x_shape[1]

        tpb = 256

        labels_dtype = classes.dtype

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
                            sample_weight,
                            sample_weight.shape[0] > 0,
                            n_classes,
                            False))

        count_classes = count_classes_kernel(x_coo_data.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / tpb), ), (tpb, ),
                      (class_c, n_rows, Y))

        self.feature_count_ = self.feature_count_ + counts
        self.class_count_ = self.class_count_ + class_c


class MultinomialNB(_BaseDiscreteNB):

    # TODO: Make this extend cuml.Base:
    # https://github.com/rapidsai/cuml/issues/1834

    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification
    with discrete features (e.g., word counts for text classification).

    The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.

    NOTE: While cuML only provides the multinomial version currently, the
    other variants are planned to be included soon. Refer to the
    corresponding Github issue for updates:
    https://github.com/rapidsai/cuml/issues/1666

    Parameters
    ----------

    alpha : float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no
        smoothing).
    fit_prior : boolean
        Whether to learn class prior probabilities or no. If false, a
        uniform prior will be used.
    class_prior : array-like, size (n_classes)
        Prior probabilities of the classes. If specified, the priors are
        not adjusted according to the data.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
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

    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    .. code-block:: python

    import cupy as cp

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

    X = cp.sparse.csr_matrix(features.tocsr(), dtype=cp.float32)
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
    def __init__(self, *,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None,
                 output_type=None,
                 handle=None,
                 verbose=False):
        super(MultinomialNB, self).__init__(handle=handle,
                                            output_type=output_type,
                                            verbose=verbose)
        self.alpha = alpha
        self.fit_prior = fit_prior

        if class_prior is not None:
            self.class_prior, *_ = input_to_cuml_array(class_prior)
        else:
            self.class_prior = None

        self.fit_called_ = False
        self.n_classes_ = 0
        self.n_features_ = None

        # Needed until Base no longer assumed cumlHandle
        self.handle = None

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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
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
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    class_log_prior_ : ndarray of shape (n_classes)
        Log probability of each class (smoothed).
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_ : int
        Number of features of each sample.
    Examples
    --------
    >>> import cupy as cp
    >>> rng = cp.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
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
    def __init__(self, *, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None, output_type=None, handle=None,
                 verbose=False):
        super(BernoulliNB, self).__init__(handle=handle,
                                          output_type=output_type,
                                          verbose=verbose)
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        if class_prior is not None:
            self.class_prior, *_ = input_to_cuml_array(class_prior)
        else:
            self.class_prior = None
        self.n_classes_ = 0
        self.n_features_ = None
        self.fit_called_ = False
        self.handle = None

    def _check_X(self, X):
        X = super()._check_X(X)
        if self.binarize is not None:
            if cp.sparse.isspmatrix(X):
                X.data = binarize(X.data, threshold=self.binarize)
            else:
                X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if self.binarize is not None:
            if cp.sparse.isspmatrix(X):
                X.data = binarize(X.data, threshold=self.binarize)
            else:
                X = binarize(X, threshold=self.binarize)
        return X, y

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

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
        self.feature_log_prob_ = (cp.log(smoothed_fc) -
                                  cp.log(smoothed_cc.reshape(-1, 1)))

    def get_param_names(self):
        return super().get_param_names() + \
            [
                "alpha",
                "binarize",
                "fit_prior",
                "class_prior",
            ]
