#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cuml.linear_model.elastic_net import ElasticNet


class Lasso(ElasticNet):

    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection and improves the conditioning of the problem.

    cuML's Lasso can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or `__cuda_array_interface__`
    compliant), in addition to cuDF objects. It uses coordinate descent to fit
    a linear model.

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>> import cudf
        >>> from cuml.linear_model import Lasso
        >>> ls = Lasso(alpha = 0.1)
        >>> X = cudf.DataFrame()
        >>> X['col1'] = np.array([0, 1, 2], dtype = np.float32)
        >>> X['col2'] = np.array([0, 1, 2], dtype = np.float32)
        >>> y = cudf.Series( np.array([0.0, 1.0, 2.0], dtype = np.float32) )
        >>> result_lasso = ls.fit(X, y)
        >>> print(result_lasso.coef_)
        0   0.85
        1   0.00
        dtype: float32
        >>> print(result_lasso.intercept_)
        0.149999...

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = np.array([3,2], dtype = np.float32)
        >>> X_new['col2'] = np.array([5,5], dtype = np.float32)
        >>> preds = result_lasso.predict(X_new)
        >>> print(preds)
        0   2.70
        1   1.85
        dtype: float32

    Parameters
    -----------
    alpha : float (default = 1.0)
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    fit_intercept : boolean (default = True)
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by the
        column-wise standard deviation.
        If False, no scaling will be done.
        Note: this is in contrast to sklearn's deprecated `normalize` flag,
        which divides by the column-wise L2 norm; but this is the same as if
        using sklearn's StandardScaler.
    max_iter : int (default = 1000)
        The maximum number of iterations
    tol : float (default = 1e-3)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    solver : {'cd', 'qn'} (default='cd')
        Choose an algorithm:

          * 'cd' - coordinate descent
          * 'qn' - quasi-newton

        You may find the alternative 'qn' algorithm is faster when the number
        of features is sufficiently large, but the sample size is small.
    selection : {'cyclic', 'random'} (default='cyclic')
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to 'random') often leads to significantly faster
        convergence especially when tol is higher than 1e-4.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """

    def __init__(self, *, alpha=1.0, fit_intercept=True,
                 normalize=False, max_iter=1000, tol=1e-3,
                 solver='cd', selection='cyclic',
                 handle=None, output_type=None, verbose=False):
        # Lasso is just a special case of ElasticNet
        super().__init__(
            l1_ratio=1.0, alpha=alpha, fit_intercept=fit_intercept,
            normalize=normalize, max_iter=max_iter, tol=tol,
            solver=solver, selection=selection,
            handle=handle, output_type=output_type, verbose=verbose)

    def get_param_names(self):
        return list(set(super().get_param_names()) - {'l1_ratio'})
