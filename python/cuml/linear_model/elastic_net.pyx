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

from cuml.solvers import CD


class ElasticNet:

    """
    ElasticNet extends LinearRegression with combined L1 and L2 regularizations
    on the coefficients when predicting response y with a linear combination of
    the predictors in X. It can reduce the variance of the predictors, force
    some coefficients to be smaell, and improves the conditioning of the
    problem.

    cuML's ElasticNet expects a cuDF DataFrame, uses coordinate descent to fit
    a linear model.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.linear_model import ElasticNet

        enet = ElasticNet(alpha = 0.1, l1_ratio=0.5)

        X = cudf.DataFrame()
        X['col1'] = np.array([0, 1, 2], dtype = np.float32)
        X['col2'] = np.array([0, 1, 2], dtype = np.float32)

        y = cudf.Series( np.array([0.0, 1.0, 2.0], dtype = np.float32) )

        result_enet = enet.fit(X, y)
        print("Coefficients:")
        print(result_enet.coef_)
        print("intercept:")
        print(result_enet.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_enet.predict(X_new)

        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 0.448408
                    1 0.443341

        Intercept:
                    0.1082506

        Preds:

                    0 3.67018
                    1 3.22177

    Parameters
    -----------
    alpha : float or double
        Constant that multiplies the L1 term. Defaults to 1.0.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept : boolean (default = True)
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2
        norm.
        If False, no scaling will be done.
    max_iter : int
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    selection : str, default ‘cyclic’
        If set to ‘random’, a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to ‘random’) often leads to significantly faster
        convergence especially when tol is higher than 1e-4.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If fit_intercept_ is False, will be 0.


    For additional docs, see `scikitlearn's ElasticNet
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, max_iter=1000, tol=1e-3, selection='cyclic'):

        """
        Initializes the elastic-net regression class.

        Parameters
        ----------
        alpha : float or double.
        l1_ratio : float or double.
        fit_intercept: boolean.
        normalize: boolean.
        max_iter: int
        tol: float or double.
        selection : str, ‘cyclic’, or 'random'

        For additional docs, see `scikitlearn's ElasticNet
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.
        """
        self._check_alpha(alpha)
        self._check_l1_ratio(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.cuElasticNet = None
        if selection in ['cyclic', 'random']:
            self.selection = selection
        else:
            msg = "selection {!r} is not supported"
            raise TypeError(msg.format(selection))

        self.intercept_value = 0.0

    def _check_alpha(self, alpha):
        if alpha <= 0.0:
            msg = "alpha value has to be positive"
            raise ValueError(msg.format(alpha))

    def _check_l1_ratio(self, l1_ratio):
        if l1_ratio < 0.0 or l1_ratio > 1.0:
            msg = "l1_ratio value has to be between 0.0 and 1.0"
            raise ValueError(msg.format(l1_ratio))

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        shuffle = False
        if self.selection == 'random':
            shuffle = True

        self.cuElasticNet = CD(fit_intercept=self.fit_intercept,
                               normalize=self.normalize, alpha=self.alpha,
                               l1_ratio=self.l1_ratio, shuffle=shuffle,
                               max_iter=self.max_iter)
        self.cuElasticNet.fit(X, y)

        self.coef_ = self.cuElasticNet.coef_
        self.intercept_ = self.cuElasticNet.intercept_

        return self

    def predict(self, X):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        return self.cuElasticNet.predict(X)

    def get_params(self, deep=True):
        """
        Sklearn style return parameter state

        Parameters
        -----------
        deep : boolean (default = True)
        """
        params = dict()
        variables = ['alpha', 'fit_intercept', 'normalize', 'max_iter', 'tol',
                     'selection']
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Sklearn style set parameter state to dictionary of params.

        Parameters
        -----------
        params : dict of new params
        """
        if not params:
            return self
        variables = ['alpha', 'fit_intercept', 'normalize', 'max_iter', 'tol',
                     'selection']
        for key, value in params.items():
            if key not in variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)

        return self
