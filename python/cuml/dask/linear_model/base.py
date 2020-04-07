# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cuml.dask.common.comms import CommsContext
from cuml.dask.common.input_utils import DistributedDataHandler
from dask.distributed import wait


class BaseLinearModelSyncFitMixin(object):

    def _fit(self, model_func, data, **kwargs):

        for d in data:
            d = self.client.persist(data)

        data = DistributedDataHandler.create(data=data, client=self.client)
        self.datatype = data.datatype

        comms = CommsContext(comms_p2p=False, verbose=self.verbose)
        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)
        self.ranks = data.ranks

        n_cols = d[0].shape[1]

        lin_models = dict([(data.worker_info[wf[0]]["r"], self.client.submit(
            model_func,
            comms.sessionId,
            self.datatype,
            **self.kwargs,
            pure=False,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])

        lin_fit = dict([(wf[0], self.client.submit(
            _func_fit,
            lin_models[data.worker_info[wf[0]]["r"]],
            wf[1],
            data.total_rows,
            n_cols,
            data.parts_to_sizes[data.worker_info[wf[0]]["r"]],
            data.worker_info[wf[0]]["r"],
            pure=False,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])

        wait(list(lin_fit.values()))
        raise_exception_from_futures(list(lin_fit.values()))

        comms.destroy()
        return lin_models


def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
    return f.fit(data, n_rows, n_cols, partsToSizes, rank)
