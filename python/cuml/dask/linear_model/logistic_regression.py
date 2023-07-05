# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedPredictionMixin
from cuml.dask.common.base import mnmg_import
from cuml.dask.common.base import SyncFitMixinLinearModel
from raft_dask.common.comms import get_raft_comm_state
from dask.distributed import get_worker

from cuml.dask.common import parts_to_ranks
from cuml.dask.common.input_utils import DistributedDataHandler, concatenate
from raft_dask.common.comms import Comms
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")

## TODO, replace this class with existing SyncFitMixnLinearModel. 
## This requires moving num_classes calculation to c++ fit
#class SyncFitMixinLogisticModel(SyncFitMixinLinearModel):
#
#    @staticmethod
#    @mnmg_import
#    def _func_fit(sessionId, objs, datatype, has_weights, **kwargs):
#        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
#
#        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
#
#        if not has_weights:
#            inp_data = concatenate(objs)
#            inp_weights = None
#        else:
#            inp_data = concatenate([X for X, weights in objs])
#            inp_weights = concatenate([weights for X, weights in objs])
#
#        return cumlKMeans(handle=handle, output_type=datatype, **kwargs).fit(
#            inp_data, sample_weight=inp_weights
#        )
#
#def _func_fit_lrmg(f, data, n_rows, n_cols, partsToSizes, rank):
#            if not has_weights:
#            inp_data = concatenate(objs)
#            inp_weights = None
#        else:
#            inp_data = concatenate([X for X, weights in objs])
#            inp_weights = concatenate([weights for X, weights in objs])
#
#    int n_ranks = partsToSizes
#    return f.fit(data, n_rows, n_cols, partsToSizes, rank)

    
class LogisticRegression(
    BaseEstimator, SyncFitMixinLinearModel 
):

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

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

        models = self._fit(
            model_func=LogisticRegression._create_model, 
            data=(X, y)
        )

        self._set_internal_model(models[0])

        return self

    def get_param_names(self):
        return list(self.kwargs.keys())

    @staticmethod
    @mnmg_import
    def _create_model(sessionId, datatype, **kwargs):
        from cuml.linear_model.logistic_regression_mg import LogisticRegressionMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return LogisticRegressionMG(
            handle=handle 
        )

    @staticmethod
    def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
        print("using logisticregression func_fit")
        print(f"type(data) is {type(data)}")
        print(f"len(data) is {len(data)}")
        print(f"(data[0][0]) is {data[0][0]}")
        print(f"(data[0][1]) is {data[0][1]}")
        inp_X = concatenate([X for X, _ in data])
        inp_y = concatenate([y for _, y in data])
        return f.fit([(inp_X, inp_y)], n_rows, n_cols, partsToSizes, rank)
