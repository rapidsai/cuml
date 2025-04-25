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

from dask.distributed import get_worker, wait
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask.common import parts_to_ranks, raise_exception_from_futures
from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.input_utils import DistributedDataHandler, to_output
from cuml.dask.common.part_utils import flatten_grouped_results


class BaseDecomposition(BaseEstimator):
    def __init__(self, *, model_func, client=None, verbose=False, **kwargs):
        """
        Constructor for distributed decomposition model
        """
        super().__init__(client=client, verbose=verbose, **kwargs)
        self._model_func = model_func


class DecompositionSyncFitMixin(object):
    @staticmethod
    def _func_fit(m, dfs, M, N, partsToRanks, rank, _transform):
        return m.fit(dfs, M, N, partsToRanks, rank, _transform)

    def _fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input

        """

        n_cols = X.shape[1]

        data = DistributedDataHandler.create(data=X, client=self.client)
        self.datatype = data.datatype

        if "svd_solver" in self.kwargs and self.kwargs["svd_solver"] == "tsqr":
            comms = Comms(comms_p2p=True)
        else:
            comms = Comms(comms_p2p=False)

        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)

        worker_info = comms.worker_info(comms.worker_addresses)
        parts_to_sizes, _ = parts_to_ranks(
            self.client, worker_info, data.gpu_futures
        )

        total_rows = data.total_rows

        models = dict(
            [
                (
                    data.worker_info[wf[0]]["rank"],
                    self.client.submit(
                        self._create_model,
                        comms.sessionId,
                        self._model_func,
                        self.datatype,
                        **self.kwargs,
                        pure=False,
                        workers=[wf[0]],
                    ),
                )
                for idx, wf in enumerate(data.worker_to_parts.items())
            ]
        )

        pca_fit = dict(
            [
                (
                    wf[0],
                    self.client.submit(
                        DecompositionSyncFitMixin._func_fit,
                        models[data.worker_info[wf[0]]["rank"]],
                        wf[1],
                        total_rows,
                        n_cols,
                        parts_to_sizes,
                        data.worker_info[wf[0]]["rank"],
                        _transform,
                        pure=False,
                        workers=[wf[0]],
                    ),
                )
                for idx, wf in enumerate(data.worker_to_parts.items())
            ]
        )

        wait(list(pca_fit.values()))
        raise_exception_from_futures(list(pca_fit.values()))

        comms.destroy()

        self._set_internal_model(list(models.values())[0])

        if _transform:
            out_futures = flatten_grouped_results(
                self.client, data.gpu_futures, pca_fit
            )
            return to_output(out_futures, self.datatype)

        return self

    @staticmethod
    def _create_model(sessionId, model_func, datatype, **kwargs):
        dask_worker = get_worker()
        handle = get_raft_comm_state(sessionId, dask_worker)["handle"]
        return model_func(handle, datatype, **kwargs)
