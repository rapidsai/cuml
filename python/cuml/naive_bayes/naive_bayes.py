#
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
#


import math

import cupy as cp
import cupy.prof
from cuml.prims.label import make_monotonic, check_labels, invert_labels

"""
A simple reduction kernel that takes in a sparse (COO) array
of features and computes the sum and sum squared for each class
label 
"""
count_features_coo = cp.RawKernel(r'''
extern "C" __global__
void count_features_coo(float *out,
                    int *rows, int *cols,
                    float *vals, int nnz,
                    int n_rows, int n_cols,
                    int *labels,
                    int n_classes,
                    bool square) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= nnz) return;
  
  int row = rows[i];
  int col = cols[i];
  float val = vals[i];
  
  if(square) val *= val;
  
  int label = labels[row];
  
  atomicAdd(out + ((col * n_classes) + label), val);
}
''', 'count_features_coo')

count_features_dense = cp.RawKernel(r'''
extern "C" __global__
void count_features_dense(
                    float *out,
                    float *in,
                    int n_rows, 
                    int n_cols,
                    int *labels,
                    int n_classes,
                    bool square,
                    bool rowMajor) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(row >= n_rows || col >= n_cols) return;

  float val = !rowMajor ? in[col * n_rows + row] : in[row * n_cols + col];
  
  if(val == 0.0) return;
  
  if(square) val *= val;
  int label = labels[row];

  atomicAdd(out + ((col * n_classes) + label), val);
}
''', 'count_features_dense')


class GaussianNB(object):

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        """FIt Gaussian Naive Bayes according to X, y

        """
        self.epsilon_ = self.var_smoothing * cp.var(X, axis=0).max()

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.theta_ = cp.zeros((n_classes, n_features))
        self.sigma_ = cp.zeros((n_classes, n_features))

        self.class_count_ = cp.zeros(n_classes)

        if self.priors is not None:
            self.class_prior_ = self.priors
        else:
            self.class_prior_ = cp.zeros(n_classes, dtype=cp.float32)

    def _count(self, X, Y):

        """
        :param X: cupy.sparse matrix of size (n_rows, n_features)
        :param Y: cupy.array of monotonic class labels
        """
        x_coo = X.tocoo()

        counts = cp.zeros((self.n_classes_, self.n_features_),
                          order="F", dtype=cp.float32)

        sq_counts = cp.zeros((self.n_classes_, self.n_features_),
                             order="F", dtype=cp.float32)

        count_features_coo((math.ceil(x_coo.nnz / 32),), (32,),
                       (counts,
                        x_coo.row,
                        x_coo.col,
                        x_coo.data,
                        x_coo.nnz,
                        x_coo.shape[0],
                        x_coo.shape[1],
                        Y, Y.shape[0],
                        self.n_classes_,
                        False))

        count_features_coo((math.ceil(x_coo.nnz / 32),), (32,),
                       (counts,
                        x_coo.row,
                        x_coo.col,
                        x_coo.data,
                        x_coo.nnz,
                        x_coo.shape[0],
                        x_coo.shape[1],
                        Y, Y.shape[0],
                        self.n_classes_,
                        True))

        self.theta_ += counts
        self.sigma_ += sq_counts
        self.class_count_ += counts.sum(axis=1).reshape(self.n_classes_)


class MultinomialNB(object):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):

        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.fit_called_ = False

        self.classes_ = None
        self.n_classes_ = 0

        self.n_features_ = None

    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    def fit(self, X, y, sample_weight=None):
        return self.partial_fit(X, y, sample_weight)

    @cp.prof.TimeRangeDecorator(message="fit()", color_id=0)
    def _partial_fit(self, X, y, sample_weight=None, _classes=None):

        Y, label_classes = make_monotonic(y, copy=True)

        if not self.fit_called_:
            self.fit_called_ = True
            if _classes is not None:
                check_labels(Y, _classes)
                self.classes_ = _classes
            else:
                self.classes_ = label_classes

            self.n_classes_ = self.classes_.shape[0]
            self.n_features_ = X.shape[1]
            self._init_counters(self.n_classes_, self.n_features_, dtype=X.dtype)
        else:
            check_labels(Y, self.classes_)

        self._count(X, Y)

        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incremental fit on a batch of samples

        :param X:
        :param y:
        :param classes:
        :param sample_weight:
        :return:
        """
        return self._partial_fit(X, y, sample_weight=sample_weight, _classes=classes)

    @cp.prof.TimeRangeDecorator(message="predict()", color_id=1)
    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1)

        print(str(self.classes_))

        y_hat = invert_labels(indices, classes=self.classes_)
        return y_hat

    def _init_counters(self, n_effective_classes, n_features, dtype=cp.float32):
        self.class_count_ = cp.zeros(n_effective_classes, order="F", dtype=dtype)
        self.feature_count_ = cp.zeros((n_effective_classes, n_features),
                                       order="F", dtype=dtype)

    def _count(self, X, Y):
        """
        :param X: cupy.sparse matrix of size (n_rows, n_features)
        :param Y: cupy.array of monotonic class labels
        """

        if X.ndim != 2:
            raise ValueError("Input samples should be a 2D array")

        counts = cp.zeros((self.n_classes_, self.n_features_),
                          order="F", dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if cp.sparse.isspmatrix(X):
            X = X.tocoo()

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

        self.feature_count_ += counts

        print(str(counts.dtype))
        self.class_count_ += counts.sum(axis=1).reshape(self.n_classes_)

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self.n_classes:
                raise ValueError("Number of classes must match number of priors")

            self.class_log_prior_ = cp.log(class_prior)

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)
            self.class_log_prior_ = log_class_count - \
                                    cp.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = cp.full(self.n_classes_,
                                            -math.log(self.n_classes_))

    def _update_feature_log_prob(self, alpha):

        """ apply add-lambda smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = (cp.log(smoothed_fc) - cp.log(smoothed_cc))

    def _joint_log_likelihood(self, X):
        """ Calculate the posterior log probability of the samples X """
        ret = X.dot(self.feature_log_prob_.T)
        ret += self.class_log_prior_.T
        return ret
