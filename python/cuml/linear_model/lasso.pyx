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


class Lasso:

    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection, and improves the conditioning of the problem.

    cuML's Lasso an array-like object or cuDF DataFrame, and
    uses coordinate descent to fit a linear model.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.linear_model import Lasso

        ls = Lasso(alpha = 0.1)

        X = cudf.DataFrame()
        X['col1'] = np.array([0, 1, 2], dtype = np.float32)
        X['col2'] = np.array([0, 1, 2], dtype = np.float32)

        y = cudf.Series( np.array([0.0, 1.0, 2.0], dtype = np.float32) )

        result_lasso = ls.fit(X, y)
        print("Coefficients:")
        print(result_lasso.coef_)
        print("intercept:")
        print(result_lasso.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_lasso.predict(X_new)

        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 0.85
                    1 0.0

        Intercept:
                    0.149999

        Preds:

                    0 2.7
                    1 1.85

    Parameters
    -----------
    alpha : float or double
        Constant that multiplies the L1 term. Defaults to 1.0.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression class.
        For numerical reasons, using alpha = 0 with the Lasso class is not
        advised.
        Given this, you should use the LinearRegression class.
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

    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 max_iter=1000, tol=1e-3, selection='cyclic'):

        """
        Initializes the lasso regression class.

        Parameters
        ----------
        alpha : float or double.
        fit_intercept: boolean.
        normalize: boolean.
        max_iter: int
        tol: float or double.
        selection : str, ‘cyclic’, or 'random'

        For additional docs, see `scikitlearn's Lasso
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
        """
        self._check_alpha(alpha)
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.culasso = None
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

        convert_dtype : bool (default = False)
            When set to True, the transform method will automatically convert
            y to be the same data type as X if they differ. This
            will increase memory used for the method.

        """

        shuffle = False
        if self.selection == 'random':
            shuffle = True

        self.culasso = CD(fit_intercept=self.fit_intercept,
                          normalize=self.normalize, alpha=self.alpha,
                          l1_ratio=1.0, shuffle=shuffle,
                          max_iter=self.max_iter)
        self.culasso.fit(X, y, convert_dtype=convert_dtype)

        self.coef_ = self.culasso.coef_
        self.intercept_ = self.culasso.intercept_

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

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        return self.culasso.predict(X, convert_dtype=convert_dtype)

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
