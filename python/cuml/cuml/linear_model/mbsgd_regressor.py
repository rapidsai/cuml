#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cuml.internals
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.solvers import SGD


class MBSGDRegressor(Base, RegressorMixin, FMajorInputTagMixin):
    """
    Linear regression model fitted by minimizing a
    regularized empirical loss with mini-batch SGD.
    The MBSGD Regressor implementation is experimental and and it uses a
    different algorithm than sklearn's SGDClassifier. In order to improve
    the results obtained from cuML's MBSGD Regressor:

    * Reduce the batch size
    * Increase the eta0
    * Increase the number of iterations

    Since cuML is analyzing the data in batches using a small eta0 might
    not let the model learn as much as scikit learn does. Furthermore,
    decreasing the batch size might seen an increase in the time required
    to fit the model.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf
        >>> from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype = cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype = cp.float32)
        >>> y = cudf.Series(cp.array([1, 1, 2, 2], dtype=cp.float32))
        >>> pred_data = cudf.DataFrame()
        >>> pred_data['col1'] = cp.asarray([3, 2], dtype=cp.float32)
        >>> pred_data['col2'] = cp.asarray([5, 5], dtype=cp.float32)
        >>> cu_mbsgd_regressor = cumlMBSGDRegressor(learning_rate='constant',
        ...                                         eta0=0.05, epochs=2000,
        ...                                         fit_intercept=True,
        ...                                         batch_size=1, tol=0.0,
        ...                                         penalty='l2',
        ...                                         loss='squared_loss',
        ...                                         alpha=0.5)
        >>> cu_mbsgd_regressor.fit(X, y)
        MBSGDRegressor()
        >>> print("cuML intercept : ", cu_mbsgd_regressor.intercept_)
        cuML intercept :  0.725...
        >>> print("cuML coef : ", cu_mbsgd_regressor.coef_)
        cuML coef :  0    0.273...
        1     0.182...
        dtype: float32
        >>> cu_pred = cu_mbsgd_regressor.predict(pred_data)
        >>> print("cuML predictions : ", cu_pred)
        cuML predictions :  0    2.456...
        1    2.183...
        dtype: float32

    Parameters
    ----------
    loss : 'squared_loss' (default = 'squared_loss')
       'squared_loss' uses linear regression
    penalty : {'l1', 'l2', 'elasticnet', None} (default = 'l2')
        The penalty (aka regularization term) to apply.

        - 'l1': L1 norm (Lasso) regularization
        - 'l2': L2 norm (Ridge) regularization (the default)
        - 'elasticnet': Elastic Net regularization, a weighted average of L1 and L2
        - None: no penalty is added

    alpha : float (default = 0.0001)
       The constant value which decides the degree of regularization
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    l1_ratio : float (default=0.15)
        The l1_ratio is used only when `penalty = elasticnet`. The value for
        l1_ratio should be `0 <= l1_ratio <= 1`. When `l1_ratio = 0` then the
        `penalty = 'l2'` and if `l1_ratio = 1` then `penalty = 'l1'`
    batch_size : int (default = 32)
        It sets the number of samples that will be included in each batch.
    epochs : int (default = 1000)
        The number of times the model should iterate through the entire dataset
        during training (default = 1000)
    tol : float (default = 1e-3)
       The training process will stop if current_loss > previous_loss - tol
    shuffle : boolean (default = True)
       True, shuffles the training data after each epoch
       False, does not shuffle the training data after each epoch
    eta0 : float (default = 0.001)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    learning_rate : {'optimal', 'constant', 'invscaling', 'adaptive'} \
        (default = 'constant')

        `optimal` option will be supported in a future version

        `constant` keeps the learning rate constant

        `adaptive` changes the learning rate if the training loss or the
        validation accuracy does not improve for `n_iter_no_change` epochs.
        The old learning rate is generally divided by 5
    n_iter_no_change : int (default = 5)
        the number of epochs to train without any improvement in the model
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    For additional docs, see `scikitlearn's SGDRegressor
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html>`_.
    """

    def __init__(
        self,
        *,
        loss="squared_loss",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        epochs=1000,
        tol=1e-3,
        shuffle=True,
        learning_rate="constant",
        eta0=0.001,
        power_t=0.5,
        batch_size=32,
        n_iter_no_change=5,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        if loss in ["squared_loss"]:
            self.loss = loss
        else:
            msg = "loss {!r} is not supported"
            raise TypeError(msg.format(loss))

        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.solver_model = SGD(**self.get_params())

    @generate_docstring()
    def fit(self, X, y, *, convert_dtype=True) -> "MBSGDRegressor":
        """
        Fit the model with X and y.

        """
        self.solver_model.fit(X, y, convert_dtype=convert_dtype)
        return self

    @property
    def dtype(self):
        return self.solver_model.dtype

    @property
    def coef_(self) -> CumlArray:
        return self.solver_model.coef_

    @coef_.setter
    def coef_(self, value):
        self.solver_model.coef_ = value

    @property
    def intercept_(self) -> float:
        return self.solver_model.intercept_

    @intercept_.setter
    def intercept_(self, value):
        self.solver_model.intercept_ = value

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @cuml.internals.api_base_return_array_skipall
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """

        preds = self.solver_model.predict(X, convert_dtype=convert_dtype)
        return preds

    def set_params(self, **params):
        super().set_params(**params)
        self.solver_model.set_params(**params)
        return self

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "loss",
            "penalty",
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "epochs",
            "tol",
            "shuffle",
            "learning_rate",
            "eta0",
            "power_t",
            "batch_size",
            "n_iter_no_change",
        ]
