#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.linear_model.elastic_net import ElasticNet


class Lasso(ElasticNet):
    """
    Linear Model trained with L1 prior as regularizer (aka the Lasso).

    This is the same as ``ElasticNet(l1_ratio=1.0)`` (no L2 penalty).

    Parameters
    ----------
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
    normalize : boolean, default=False

        .. deprecated:: 25.12
            ``normalize`` is deprecated and will be removed in 26.02. When
            needed, please use a ``StandardScaler`` to normalize your data
            before passing to ``fit``.

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.

    Examples
    --------
    >>> import numpy as np
    >>> import cudf
    >>> from cuml.linear_model import Lasso
    >>> ls = Lasso(alpha = 0.1, solver='qn')
    >>> X = cudf.DataFrame()
    >>> X['col1'] = np.array([0, 1, 2], dtype = np.float32)
    >>> X['col2'] = np.array([0, 1, 2], dtype = np.float32)
    >>> y = cudf.Series( np.array([0.0, 1.0, 2.0], dtype = np.float32) )
    >>> result_lasso = ls.fit(X, y)
    >>> print(result_lasso.coef_)
    0   0.425
    1   0.425
    dtype: float32
    >>> print(result_lasso.intercept_)
    0.150000...

    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = np.array([3,2], dtype = np.float32)
    >>> X_new['col2'] = np.array([5,5], dtype = np.float32)
    >>> preds = result_lasso.predict(X_new)
    >>> print(preds)
    0   3.549997
    1   3.124997
    dtype: float32
    """

    _cpu_class_path = "sklearn.linear_model.Lasso"

    @classmethod
    def _get_param_names(cls):
        return list(set(super()._get_param_names()) - {"l1_ratio"})

    @classmethod
    def _params_from_cpu(cls, model):
        out = super()._params_from_cpu(model)
        out.pop("l1_ratio")
        return out

    def _params_to_cpu(self):
        out = super()._params_to_cpu()
        out.pop("l1_ratio")
        return out

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        solver="cd",
        selection="cyclic",
        normalize=False,
        handle=None,
        output_type=None,
        verbose=False,
    ):
        # Lasso is just a special case of ElasticNet
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            selection=selection,
            normalize=normalize,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
        )
