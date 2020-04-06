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

from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.comms import worker_state, CommsContext

from cuml.dask.common.input_utils import to_output

from cuml.dask.common.part_utils import flatten_grouped_results

from dask.distributed import wait

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.input_utils import DistributedDataHandler


class BaseDecomposition(BaseEstimator):

    def __init__(self, model_func, client=None, verbose=False, **kwargs):
        """
        Constructor for distributed decomposition model
        """
        super(BaseDecomposition, self).__init__(client=client,
                                                verbose=verbose,
                                                **kwargs)
        self._model_func = model_func

        # define attributes to make sure they
        # are available even on untrained object
        self.local_model = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None


class DecompositionSyncFitMixin(object):

    @staticmethod
    def _func_fit(m, dfs, M, N, partsToRanks, rank, transform):
        return m.fit(dfs, M, N, partsToRanks, rank, transform)

    def _fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input

        """

        X = self.client.persist(X)

        data = DistributedDataHandler.create(data=X, client=self.client)
        self.datatype = data.datatype

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)

        total_rows = data.total_rows
        n_cols = X.shape[1]

        models = dict([(data.worker_info[wf[0]]["r"], self.client.submit(
            self._create_model,
            comms.sessionId,
            self._model_func,
            self.datatype,
            **self.kwargs,
            pure=False,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])

        pca_fit = dict([(wf[0], self.client.submit(
            DecompositionSyncFitMixin._func_fit,
            models[data.worker_info[wf[0]]["r"]],
            wf[1],
            total_rows, n_cols,
            data.parts_to_sizes[data.worker_info[wf[0]]["r"]],
            data.worker_info[wf[0]]["r"],
            _transform,
            pure=False,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])

        wait(list(pca_fit.values()))
        raise_exception_from_futures(list(pca_fit.values()))

        comms.destroy()

        self.local_model = list(models.values())[0].result()

        self.components_ = self.local_model.components_
        self.explained_variance_ = self.local_model.explained_variance_
        self.explained_variance_ratio_ = \
            self.local_model.explained_variance_ratio_
        self.singular_values_ = self.local_model.singular_values_

        if _transform:
            out_futures = flatten_grouped_results(self.client,
                                                  data.gpu_futures,
                                                  pca_fit)
            return to_output(out_futures, self.datatype)

        return self

    @staticmethod
    def _create_model(sessionId, model_func, datatype, **kwargs):
        handle = worker_state(sessionId)["handle"]
        return model_func(handle, datatype, **kwargs)
