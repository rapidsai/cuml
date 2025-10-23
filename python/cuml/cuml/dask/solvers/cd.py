# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from dask.distributed import get_worker
from raft_dask.common.comms import get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    SyncFitMixinLinearModel,
    mnmg_import,
)


class CD(BaseEstimator, SyncFitMixinLinearModel, DelayedPredictionMixin):
    """
    Model-Parallel Multi-GPU Linear Regression Model.
    """

    def __init__(self, *, client=None, **kwargs):
        """
        Initializes the linear regression class.

        """
        super().__init__(client=client, **kwargs)
        self._model_fit = False
        self._consec_call = 0

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Features for regression
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
            Labels (outcome values)
        """

        models = self._fit(model_func=CD._create_model, data=(X, y))

        self._set_internal_model(list(models.values())[0])

        return self

    def predict(self, X, delayed=True):
        """
        Make predictions for X and returns a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
        """
        return self._predict(X, delayed=delayed)

    @staticmethod
    @mnmg_import
    def _create_model(sessionId, datatype, **kwargs):
        from cuml.solvers.cd_mg import CDMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return CDMG(handle=handle, output_type=datatype, **kwargs)
