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


import cudf
import numpy as np
from cuml.solvers import CD

class Lasso:

    """
    Ridge extends LinearRegression by providing L2 regularization on the coefficients when
    predicting response y with a linear combination of the predictors in X. It can reduce
    the variance of the predictors, and improves the conditioning of the problem.

    cuML's Ridge expects a cuDF DataFrame, and provides 3 algorithms SVD, Eig and CD to
    fit a linear model. SVD is more stable, but Eig (default) is much more faster. CD uses
    Coordinate Descent and can be faster if the data is large.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf

        # Both import methods supported
        from cuml import Ridge
        from cuml.linear_model import Ridge

        alpha = np.array([1.0])
        ridge = Ridge(alpha = alpha, fit_intercept = True, normalize = False, solver = "eig")

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        result_ridge = ridge.fit(X_cudf, y_cudf)
        print("Coefficients:")
        print(result_ridge.coef_)
        print("intercept:")
        print(result_ridge.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_ridge.predict(X_new)

        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 1.0000001
                    1 1.9999998

        Intercept:
                    3.0

        Preds:

                    0 15.999999
                    1 14.999999

    Parameters
    -----------
    alpha : float or double
        Regularization strength - must be a positive float. Larger values specify
        stronger regularization. Array input will be supported later.
    solver : 'eig' or 'svd' or 'cd' (default = 'eig')
        Eig uses a eigendecomposition of the covariance matrix, and is much faster.
        SVD is slower, but is guaranteed to be stable.
        CD or Coordinate Descent is very fast and is suitable for large problems.
    fit_intercept : boolean (default = True)
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2 norm.
        If False, no scaling will be done.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If fit_intercept_ is False, will be 0.
        
    Notes
    ------
    Ridge provides L2 regularization. This means that the coefficients can shrink to become
    very very small, but not zero. This can cause issues of interpretabiliy on the coefficients.
    Consider using Lasso, or thresholding small coefficients to zero.
    
    **Applications of Ridge**
        
        Ridge Regression is used in the same way as LinearRegression, but is used more frequently
        as it does not suffer from multicollinearity issues. Ridge is used in insurance premium
        prediction, stock market analysis and much more.


    For additional docs, see `scikitlearn's Ridge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    # Link will work later
    # For an additional example see `the Ridge notebook <https://github.com/rapidsai/notebooks/blob/master/cuml/ridge.ipynb>`_.
    # New link : https://github.com/rapidsai/notebooks/blob/master/cuml/ridge_regression_demo.ipynb


    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=1e-4, selection='cyclic'):

        """
        Initializes the linear ridge regression class.

        Parameters
        ----------
        solver : Type: string. 'eig' (default) and 'svd' are supported algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

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
            raise TypeError(msg.format(alpha))

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

        self.culasso = CD(fit_intercept=self.fit_intercept, normalize=self.normalize, alpha=self.alpha, 
                          l1_ratio=1.0, shuffle=shuffle, max_iter=self.max_iter)
        self.culasso.fit(X, y)

        self.coef_ = self.culasso.coef_
        self.intercept_ = self.culasso.intercept_

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

        return self.culasso.predict(X)


    def get_params(self, deep=True):
        """
        Sklearn style return parameter state

        Parameters
        -----------
        deep : boolean (default = True)
        """
        params = dict()
        variables = ['alpha', 'fit_intercept', 'normalize', 'max_iter', 'tol', 'selection']
        for key in variables:
            var_value = getattr(self,key,None)
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
        variables = ['alpha', 'fit_intercept', 'normalize', 'max_iter', 'tol', 'selection']
        for key, value in params.items():
            if key not in variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
        
        return self
