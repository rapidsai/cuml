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

from cuml.dask.solvers import CD
from cuml.dask.common.base import BaseEstimator


class Lasso(BaseEstimator):

    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection and improves the conditioning of the problem.

    cuML's Lasso an array-like object or cuDF DataFrame and
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

    def __init__(self, client=None, verbose=False, **kwargs):
        super(Lasso, self).__init__(client=client,
                                    verbose=verbose,
                                    **kwargs)

        self.solver = CD(client=client,
                         verbose=verbose,
                         **kwargs)


    def fit(self, X, y, force_colocality=False):
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
            When set to True, the transform method will, when necessary,
            convert y to be the same data type as X if they differ. This
            will increase memory used for the method.

        """

        self.solver.fit(X, y, delayed=True)

        self.coef_ = self.solver.coef_
        self.intercept_ = self.solver.intercept_

        return self

    def predict(self, X, delayed=True):
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

        return self.solver.predict(X)
