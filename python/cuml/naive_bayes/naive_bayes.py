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
import cupy.prof
import math
import warnings

from cuml.common import logger

from cuml.common import with_cupy_rmm
from cuml.common import CumlArray
from cuml.common.base import Base
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.common.import_utils import has_scipy
from cuml.prims.label import make_monotonic
from cuml.prims.label import check_labels
from cuml.prims.label import invert_labels

from cuml.metrics import accuracy_score


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

      printf("val=%f\n", val);
      
      if(has_weights)
        val *= weights[i];
          
      if(square) val *= val;
      
      {1} label = labels[row];
      atomicAdd(out + ((col * n_classes) + label), val);
    }'''

    return cuda_kernel_factory(kernel_str,
                               (float_dtype, int_dtype),
                               "count_features_coo")


def count_classes_kernel(float_dtype, int_dtype):
    kernel_str = r'''
    ({0} *out, int n_rows, {1} *labels) {

      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if(row >= n_rows) return;
      {1} label = labels[row];
      
      printf("label=%d\n", label);
      
      atomicAdd(out + label, ({0})1);
    }'''

    return cuda_kernel_factory(kernel_str,
                               (float_dtype, int_dtype),
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

      printf("val=%f\n", val);

      if(val == 0.0) return;

      if(square) val *= val;
      
      {1} label = labels[row];
      
      {1} idx = !rowMajor ? col : row;

      atomicAdd(out + ((idx * n_classes) + label), val);
    }'''

    return cuda_kernel_factory(kernel_str,
                               (float_dtype, int_dtype,),
                               "count_features_dense")


class _BaseNB(Base):

    def __init__(self, verbose=False, output_type=None):

        super(_BaseNB, self).__init__(verbose=verbose,
                                      output_type=output_type)


    @cp.prof.TimeRangeDecorator(message="predict()", color_id=1)
    @with_cupy_rmm
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)

        Returns
        -------

        C : cupy.ndarray of shape (n_samples)

        """
        out_type = self._get_output_type(X)

        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cp.sparse.isspmatrix(X):
            X = X.tocoo()
            rows = cp.asarray(X.row, dtype=X.row.dtype)
            cols = cp.asarray(X.col, dtype=X.col.dtype)
            data = cp.asarray(X.data, dtype=X.data.dtype)
            X = cp.sparse.coo_matrix((data, (rows, cols)), shape=X.shape)
        else:
            X = input_to_cuml_array(X, order='K').array.to_output('cupy')

        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1).astype(self._classes_.dtype)

        y_hat = invert_labels(indices, classes=self._classes_)
        return CumlArray(data=y_hat).to_output(out_type)

    @with_cupy_rmm
    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)


        Returns
        -------

        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in the
            model. The columns correspond to the classes in sorted order, as
            they appear in the attribute classes_.
        """
        out_type = self._get_output_type(X)

        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cp.sparse.isspmatrix(X):
            X = X.tocoo()
            rows = cp.asarray(X.row, dtype=X.row.dtype)
            cols = cp.asarray(X.col, dtype=X.col.dtype)
            data = cp.asarray(X.data, dtype=X.data.dtype)
            X = cp.sparse.coo_matrix((data, (rows, cols)), shape=X.shape)
        else:
            X = input_to_cuml_array(X, order='K').array.to_output('cupy')

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
        return CumlArray(result).to_output(out_type)

    @with_cupy_rmm
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)

        Returns
        -------

        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the model.
            The columns correspond to the classes in sorted order, as they
            appear in the attribute classes_.
        """
        out_type = self._get_output_type(X)
        result = cp.exp(self.predict_log_proba(X))
        return CumlArray(result).to_output(out_type)


class GaussianNB(_BaseNB):

    def __init__(self, priors=None, var_smoothing=1e-9,
                 output_type=None, verbose=False):

        super(GaussianNB, self).__init__(verbose=verbose,
                                         output_type=output_type)
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.fit_called = False
        self._classes_ = None

    def fit(self, X, y, sample_weight=None):
        return self._partial_fit(X, y, classes=None, _refit=True,
                                 sample_weight=sample_weight)

    @with_cupy_rmm
    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):

        if _refit:
            self._classes_ = None

        Y, label_classes = make_monotonic(y, copy=True)

        logger.debug("LABELS: "+ str(label_classes))

        self.epsilon_ = self.var_smoothing #* cp.var(X, axis=0).max()

        if not self.fit_called:

            # Original labels are stored on the instance
            if self._classes_ is None:
                self._classes_ = label_classes

            logger.debug("self classes: " + str(self._classes_))

            n_features = X.shape[1]
            n_classes = len(self._classes_)

            self.n_classes_ = n_classes

            self.theta_ = cp.zeros((n_classes, n_features))
            self.sigma_ = cp.zeros((n_classes, n_features))

            self.class_count_ = cp.zeros(n_classes, dtype=X.dtype)

        else:
            self.sigma_[:, :] -= self.epsilon_

        unique_y = cp.unique(y)
        unique_y_in_classes = cp.in1d(unique_y, self._classes_)

        if not cp.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist "
                             "in the initial classes %s" %
                             (unique_y[~unique_y_in_classes], self._classes_))

        self.theta_, self.sigma_ = self._update_mean_variance(X, Y,
                                                              n_classes,
                                                              n_features)

        self.sigma_[:, :] += self.epsilon_

        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self._partial_fit(X, y, classes, _refit=False,
                                 sample_weight=sample_weight)
    #
    # def predict(self, X):
    #     jll = self._joint_log_likelihood(X)
    #     return self._classes_[cp.argmax(jll, axis=1)]

    def _update_mean_variance(self, X, Y, n_classes, n_features,
                              sample_weight=None):

        if sample_weight is None:
            sample_weight = cp.zeros(0)

        labels_dtype = self._classes_.dtype

        logger.debug("CC: "+ str(self.class_count_))

        mu = self.theta_
        var = self.sigma_

        early_return = self.class_count_.sum() == 0
        n_past = self.class_count_

        if X.shape[0] == 0:
            return mu, var

        logger.debug(str(Y))

        new_mu = cp.zeros((n_classes, n_features))
        new_var = cp.zeros((n_classes, n_features))
        if cp.sparse.isspmatrix(X):
            X = X.tocoo()

            count_features_coo = count_features_coo_kernel(X.dtype,
                                                           labels_dtype)

            # Run once for averages
            count_features_coo((math.ceil(X.nnz / 32),), (32,),
                               (new_mu,
                                X.row,
                                X.col,
                                X.data,
                                X.nnz,
                                X.shape[0],
                                X.shape[1],
                                Y,
                                sample_weight,
                                sample_weight.shape[0] > 0,
                                self.n_classes_, False))

            # Run again for variance
            count_features_coo((math.ceil(X.nnz / 32),), (32,),
                               (new_var,
                                X.row,
                                X.col,
                                X.data,
                                X.nnz,
                                X.shape[0],
                                X.shape[1],
                                Y,
                                sample_weight,
                                sample_weight.shape[0] > 0,
                                self.n_classes_, True))

        else:

            count_features_dense = count_features_dense_kernel(X.dtype,
                                                               labels_dtype)

            # Run once for averages
            count_features_dense((math.ceil(X.shape[0] / 32),
                                  math.ceil(X.shape[1] / 32), 1),
                                 (32, 32, 1),
                                 (new_mu,
                                  X,
                                  X.shape[0],
                                  X.shape[1],
                                  Y,
                                  sample_weight,
                                  sample_weight.shape[0] > 0,
                                  self.n_classes_,
                                  False,
                                  X.flags["C_CONTIGUOUS"]))

            # Run again for variance
            count_features_dense((math.ceil(X.shape[0] / 32),
                                  math.ceil(X.shape[1] / 32), 1),
                                 (32, 32, 1),
                                 (new_var,
                                  X,
                                  X.shape[0],
                                  X.shape[1],
                                  Y,
                                  sample_weight,
                                  sample_weight.shape[0] > 0,
                                  self.n_classes_,
                                  True,
                                  X.flags["C_CONTIGUOUS"]))

        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes((math.ceil(X.shape[0] / 32),), (32,),
                      (self.class_count_, X.shape[0], Y))

        if early_return:
            logger.debug("RETURNING EARLY")
            return new_mu, new_var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
        else:
            n_new = X.shape[0]

        class_counts = cp.expand_dims(self.class_count_, axis=1)
        new_mu /= class_counts

        logger.debug("n_past: "+ str(n_past))

        # Construct variance from sum squares
        new_var = (var / class_counts) - new_mu ** 2

        n_total = n_past + n_new

        # TODO: This is really bad...
        total_mu = ((new_mu.T + (n_new + n_past)[:, cp.newaxis].T) * mu.T).T #/ n_total

        logger.debug("total_mu: " + str(total_mu.shape))

        logger.debug("N_PAST: " + str(n_past.shape))
        logger.debug("VAR: " + str(var.shape))

        old_ssd = n_past[:, cp.newaxis] * var
        new_ssd = n_new * new_var

        ssd_sum = old_ssd + new_ssd
        combined_feature_counts = n_new * n_past / n_total
        mean_adj = (mu - new_mu)**2

        total_ssd = (ssd_sum +
                     combined_feature_counts[:, cp.newaxis] *
                     mean_adj)

        total_var = total_ssd / n_total[:, cp.newaxis]

        logger.debug("total_var: " + str(total_var.shape))

        return total_mu, total_var

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []

        for i in range(cp.size(self._classes_)):
            joint1 = cp.log(self.class_prior_[i])

            n_ij = -0.5 * cp.sum(cp.log(2. * cp.pi * self.sigma_[i, :]))

            centered = X - self.theta_[i, :] ** 2
            zvals = centered / self.sigma_[i, :]
            summed = cp.sum(zvals, axis=1)
            logger.debug("normalized: "+ str(zvals.shape))
            logger.debug("nij: "+ str(n_ij.shape))
            logger.debug("summed: " + str(summed.shape))

            n_ij = -(0.5 * summed) + n_ij
            joint_log_likelihood.append(joint1 + n_ij)

        logger.debug(str(cp.argmax(cp.array(joint_log_likelihood), axis=0)))

        return cp.array(joint_log_likelihood).T


class MultinomialNB(_BaseNB):

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
    @with_cupy_rmm
    def __init__(self,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None,
                 output_type=None,
                 handle=None):
        """
        Create new multinomial Naive Bayes instance

        Parameters
        ----------

        alpha : float Additive (Laplace/Lidstone) smoothing parameter (0 for
                no smoothing).
        fit_prior : boolean Whether to learn class prior probabilities or no.
                    If false, a uniform prior will be used.
        class_prior : array-like, size (n_classes) Prior probabilities of the
                      classes. If specified, the priors are not adjusted
                      according to the data.
        """
        super(MultinomialNB, self).__init__(handle=handle,
                                            output_type=output_type)
        self.alpha = alpha
        self.fit_prior = fit_prior

        if class_prior is not None:
            self._class_prior, *_ = input_to_cuml_array(class_prior)
        else:
            self.class_prior = None

        self.fit_called_ = False

        self._classes_ = None
        self.n_classes_ = 0

        self.n_features_ = None

        # Needed until Base no longer assumed cumlHandle
        self.handle = None

    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    @with_cupy_rmm
    def fit(self, X, y, sample_weight=None):
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

    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    @with_cupy_rmm
    def _partial_fit(self, X, y, sample_weight=None, _classes=None):
        self._set_output_type(X)

        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.common.import_utils import dummy_function_always_false \
                as scipy_sparse_isspmatrix

        # todo: use a sparse CumlArray style approach when ready
        # https://github.com/rapidsai/cuml/issues/2216
        if scipy_sparse_isspmatrix(X) or cp.sparse.isspmatrix(X):
            X = X.tocoo()
            rows = cp.asarray(X.row, dtype=X.row.dtype)
            cols = cp.asarray(X.col, dtype=X.col.dtype)
            data = cp.asarray(X.data, dtype=X.data.dtype)
            X = cp.sparse.coo_matrix((data, (rows, cols)), shape=X.shape)
        else:
            X = input_to_cuml_array(X, order='K').array.to_output('cupy')

        y = input_to_cuml_array(y).array.to_output('cupy')

        Y, label_classes = make_monotonic(y, copy=True)

        if not self.fit_called_:
            self.fit_called_ = True
            if _classes is not None:
                _classes, *_ = input_to_cuml_array(_classes, order='K')
                check_labels(Y, _classes.to_output('cupy'))
                self._classes_ = _classes
            else:
                self._classes_ = CumlArray(data=label_classes)

            self.n_classes_ = self._classes_.shape[0]
            self.n_features_ = X.shape[1]
            self._init_counters(self.n_classes_, self.n_features_,
                                X.dtype)
        else:
            check_labels(Y, self._classes_)

        self._count(X, Y)

        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    @with_cupy_rmm
    def update_log_probs(self):
        """
        Updates the log probabilities. This enables lazy update for
        applications like distributed Naive Bayes, so that the model
        can be updated incrementally without incurring this cost each
        time.
        """
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

    @with_cupy_rmm
    def partial_fit(self, X, y, classes=None, sample_weight=None):
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


    @with_cupy_rmm
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a
        harsh metric since you require for each sample that each label set be
        correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. Currently, sample weight is ignored

        Returns
        -------

        score : float Mean accuracy of self.predict(X) with respect to y.
        """
        y_hat = self.predict(X)
        return accuracy_score(y_hat, cp.asarray(y, dtype=y.dtype))

    def _init_counters(self, n_effective_classes, n_features, dtype):
        self._class_count_ = CumlArray.zeros(n_effective_classes,
                                             order="F", dtype=dtype)
        self._feature_count_ = CumlArray.zeros((n_effective_classes,
                                                n_features),
                                               order="F", dtype=dtype)

    def _count(self, X, Y):
        """
        Sum feature counts & class prior counts and add to current model.

        Parameters
        ----------
        X : cupy.ndarray or cupy.sparse matrix of size
                  (n_rows, n_features)
        Y : cupy.array of monotonic class labels
        """

        if X.ndim != 2:
            raise ValueError("Input samples should be a 2D array")

        if Y.dtype != self._classes_.dtype:
            warnings.warn("Y dtype does not match classes_ dtype. Y will be "
                          "converted, which will increase memory consumption")

        counts = cp.zeros((self.n_classes_, self.n_features_), order="F",
                          dtype=X.dtype)

        class_c = cp.zeros(self.n_classes_, order="F", dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        labels_dtype = self._classes_.dtype

        if cp.sparse.isspmatrix(X):
            X = X.tocoo()

            count_features_coo = count_features_coo_kernel(X.dtype,
                                                           labels_dtype)
            count_features_coo((math.ceil(X.nnz / 32),), (32,),
                               (counts,
                                X.row,
                                X.col,
                                X.data,
                                X.nnz,
                                n_rows,
                                n_cols,
                                Y,
                                self.n_classes_, False))

        else:

            count_features_dense = count_features_dense_kernel(X.dtype,
                                                               labels_dtype)
            count_features_dense((math.ceil(n_rows / 32),
                                  math.ceil(n_cols / 32), 1),
                                 (32, 32, 1),
                                 (counts,
                                  X,
                                  n_rows,
                                  n_cols,
                                  Y,
                                  self.n_classes_,
                                  False,
                                  X.flags["C_CONTIGUOUS"]))

        count_classes = count_classes_kernel(X.dtype, labels_dtype)
        count_classes((math.ceil(n_rows / 32),), (32,),
                      (class_c, n_rows, Y))

        self._feature_count_ += counts
        self._class_count_ += class_c

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self.n_classes:
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
        self.feature_log_prob_ = (cp.log(smoothed_fc) -
                                  cp.log(smoothed_cc.reshape(-1, 1)))

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
