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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.solvers import QN
from cuml.common.base import Base

import numpy as np
import warnings

from cuml.utils import input_to_dev_array

supported_penalties = ['l1', 'l2', 'none', 'elasticnet']

supported_solvers = ['qn', 'lbfgs', 'owl']


class LogisticRegression(Base):
    """
    LogisticRegression is a linear model that is used to model probability of
    occurrence of certain events, for example probability of success or fail of
    an event.

    cuML's LogisticRegression can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant).
    It provides both single-class (using sigmoid loss) and multiple-class
    (using softmax loss) variants, depending on the input variables.

    Only one solver option is currently available: Quasi-Newton (QN)
    algorithms. Even though it is presented as a single option, this solver
    resolves to two different algorithms underneath:

    - Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
    regularization
    - Limited Memory BFGS (L-BFGS) otherwise.

    Note that, just like in Scikit-learn, the bias will not be regularized.

    Examples
    ---------
    .. code-block:: python

        import cudf
        import numpy as np

        # Both import methods supported
        # from cuml import LogisticRegression
        from cuml.linear_model import LogisticRegression

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series( np.array([0.0, 0.0, 1.0, 1.0], dtype = np.float32) )

        reg = LogisticRegression()
        reg.fit(X,y)

        print("Coefficients:")
        print(reg.coef_.copy_to_host())
        print("Intercept:")
        print(reg.intercept_.copy_to_host())

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([1,5], dtype = np.float32)
        X_new['col2'] = np.array([2,5], dtype = np.float32)

        preds = reg.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python
        Coefficients:
                    0.22309814
                    0.21012752
        Intercept:
                    -0.7548761
        Predictions:
                    0    0.0
                    1    1.0

    Parameters
    -----------
    penalty: 'none', 'l1', 'l2', 'elasticnet' (default = 'l2')
        Used to specify the norm used in the penalization.
        If 'none' or 'l2' are selected, then L-BFGS solver will be used.
        If 'l1' is selected, solver OWL-QN will be used.
        If 'elasticnet' is selected, OWL-QN will be used if l1_ratio > 0,
        otherwise L-BFGS will be used.
    tol: float (default = 1e-4)
       The training process will stop if current_loss > previous_loss - tol
    C: float (default = 1.0)
       Inverse of regularization strength; must be a positive float.
    fit_intercept: boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    class_weight: None
        Custom class weighs are currently not supported.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    verbose: bool (optional, default False)
        Controls verbosity of logging.
    l1_ratio: float or None, optional (default=None)
        The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`
    solver: 'qn', 'lbfgs', 'owl' (default=qn).
        Algorithm to use in the optimization problem. Currently only `qn` is
        supported, which automatically selects either L-BFGS or OWL-QN
        depending on the condictions of the l1 regularization described
        above. Options 'lbfgs' and 'owl' are just convenience values that
        end up using the same solver following the same rules.

    Attributes
    -----------
    coef_: device array, shape (n_classes, n_features)
        The estimated coefficients for the linear regression model.
    intercept_: device array (n_classes, 1)
        The independent term. If fit_intercept_ is False, will be 0.

    Notes
    ------

    cuML's LogisticRegression uses a different solver that the equivalent
    Scikit-learn except when there is no penalty and `solver=lbfgs` is
    chosen in Scikit-learn. This can cause (smaller) differences in the
    coefficients and predictions of the model, similar to difference when
    using different solvers in Scikit-learn.

    For additional docs, see Scikit-learn's LogistRegression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """

    def __init__(self, penalty='l2', tol=1e-4, C=1.0, fit_intercept=True,
                 class_weight=None, max_iter=1000, verbose=0, l1_ratio=None,
                 solver='qn', handle=None):

        super(LogisticRegression, self).__init__(handle=handle, verbose=False)

        if class_weight:
            raise ValueError("`class_weight` not supported.")

        if penalty not in supported_penalties:
            raise ValueError("`penalty` " + str(penalty) + "not supported.")

        if solver not in supported_solvers:
            raise ValueError("Only quasi-newton `qn` (lbfgs and owl) solvers "
                             " supported.")

        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter = max_iter
        if self.penalty == 'elasticnet':
            if l1_ratio is None:
                raise ValueError("l1_ratio has to be specified for"
                                 "loss='elasticnet'")
            if l1_ratio < 0.0 or l1_ratio > 1.0:
                msg = "l1_ratio value has to be between 0.0 and 1.0"
                raise ValueError(msg.format(l1_ratio))
            self.l1_ratio = l1_ratio

    def fit(self, X, y, convert_dtype=False):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (floats or doubles) of shape (n_samples, 1).
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the fit method will, when necessary, convert
            y to be the same data type as X if they differ. This
            will increase memory used for the method.

        """

        # Converting y to device array here to use `unique` function
        # since calling input_to_dev_array again in QN has no cost
        # Not needed to check dtype since qn class checks it already
        y_m, _, _, _, _ = input_to_dev_array(y)

        try:
            import cupy as cp
            unique_labels = cp.unique(y_m)
        except ImportError:
            warnings.warn("Using NumPy for number of class detection,"
                          "install CuPy for faster processing.")
            unique_labels = np.unique(y_m.copy_to_host())

        num_classes = len(unique_labels)

        if num_classes > 2:
            loss = 'softmax'
        else:
            loss = 'sigmoid'

        if self.penalty == 'none':
            l1_strength = 0.0
            l2_strength = 0.0

        elif self.penalty == 'l1':
            l1_strength = 1.0 / self.C
            l2_strength = 0.0

        elif self.penalty == 'l2':
            l1_strength = 0.0
            l2_strength = 1.0 / self.C

        else:
            strength = 1.0 / self.C
            l1_strength = self.l1_ratio * strength
            l2_strength = (1.0 - self.l1_ratio) * strength

        self.qn = QN(loss=loss, fit_intercept=self.fit_intercept,
                     l1_strength=l1_strength, l2_strength=l2_strength,
                     max_iter=self.max_iter, tol=self.tol,
                     verbose=self.verbose, handle=self.handle)

        self.qn.fit(X, y_m, convert_dtype=convert_dtype)

        # coefficients and intercept are contained in the same array
        if self.fit_intercept:
            self.coef_ = self.qn.coef_[0:-1]
            self.intercept_ = self.qn.coef_[-1]
        else:
            self.coef_ = self.qn.coef_

        return self

    def predict(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """
        return self.qn.predict(X, convert_dtype=convert_dtype)
