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

from cuml.dask.common import extract_ddf_partitions, extract_colocated_ddf_partitions, workers_to_parts
from cuml.dask.common import to_dask_cudf
from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.comms import worker_state, CommsContext

from collections import OrderedDict

from dask.distributed import default_client
from dask.distributed import wait

from functools import reduce

from uuid import uuid1


class CD(object):
    """
    Model-Parallel Multi-GPU Linear Regression Model. Single Process Multi GPU
    supported currently
    """

    def __init__(self, client=None, **kwargs):
        """
        Initializes the linear regression class.

        """
        self.client = default_client() if client is None else client
        self.kwargs = kwargs
        self.coef_ = None
        self.intercept_ = None
        self._model_fit = False
        self._consec_call = 0

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.solvers.cd_mg import CDMG as cumlCD
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]
        return cumlCD(handle=handle, **kwargs)

    @staticmethod
    def _func_fit_colocated(f, data, n_rows, n_cols, partsToSizes, rank):
        return f.fit_colocated(data, n_rows, n_cols, partsToSizes, rank)

    @staticmethod
    def _func_fit(f, X, y, n_rows, n_cols, partsToSizes, rank):
        return f.fit(X, y, n_rows, n_cols, partsToSizes, rank)

    @staticmethod
    def _func_predict(f, df, n_rows, n_cols, partsToSizes, rank):
        return f.predict(df, n_rows, n_cols, partsToSizes, rank)

    @staticmethod
    def _func_get_first(f):
        return f[0]

    @staticmethod
    def _func_get_idx(f, idx):
        return f[idx]

    @staticmethod
    def _func_xform(model, df):
        return model.transform(df)

    @staticmethod
    def _func_get_size_cl(df):
        return df[0].shape[0]

    @staticmethod
    def _func_get_size(df):
        return df.shape[0]

    def _fit(self, X, y):
        X_futures = self.client.sync(extract_ddf_partitions, X)
        y_futures = self.client.sync(extract_ddf_partitions, y)

        X_partition_workers = [w for w, xc in X_futures]
        y_partition_workers = [w for w, xc in y_futures]

        if set(X_partition_workers) != set(y_partition_workers):
            raise ValueError("""
              X  and y are not partitioned on the same workers expected \n_cols
              Linear Regression""")

        self.rnks = dict()
        rnk_counter = 0
        worker_to_parts = OrderedDict()
        for w, p in X_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            if w not in self.rnks.keys():
                self.rnks[w] = rnk_counter
                rnk_counter = rnk_counter + 1
            worker_to_parts[w].append(p)

        # Future TODO: add a DefaultOrderedDict to utils
        worker_to_parts_y = OrderedDict()
        for w, p in y_futures:
            if w not in worker_to_parts_y:
                worker_to_parts_y[w] = []
            worker_to_parts_y[w].append(p)

        workers = list(map(lambda x: x[0], X_futures))

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=workers)

        worker_info = comms.worker_info(comms.worker_addresses)

        key = uuid1()
        partsToSizes = [(worker_info[wf[0]]["r"], self.client.submit(
            CD._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(X_futures)]

        n_cols = X.shape[1]
        n_rows = reduce(lambda a, b: a+b, map(lambda x: x[1], partsToSizes))

        key = uuid1()
        self.linear_models = [(wf[0], self.client.submit(
            CD._func_create_model,
            comms.sessionId,
            **self.kwargs,
            workers=[wf[0]],
            key="%s-%s" % (key, idx)))
            for idx, wf in enumerate(worker_to_parts.items())]

        key = uuid1()
        linear_fit = dict([(worker_info[wf[0]]["r"], self.client.submit(
            CD._func_fit,
            wf[1],
            worker_to_parts[wf[0]],
            worker_to_parts_y[wf[0]],
            n_rows, n_cols,
            partsToSizes,
            worker_info[wf[0]]["r"],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.linear_models)])

        wait(list(linear_fit.values()))
        raise_exception_from_futures(list(linear_fit.values()))

        comms.destroy()

        self.local_model = self.linear_models[0][1].result()
        self.coef_ = self.local_model.coef_
        self.intercept_ = self.local_model.intercept_

    def _fit_with_colocality(self, X, y):
        input_futures = self.client.sync(extract_colocated_ddf_partitions,
                                         X, y, self.client)
        workers = list(input_futures.keys())

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=workers)

        worker_info = comms.worker_info(comms.worker_addresses)

        n_cols = X.shape[1]
        n_rows = 0

        self.rnks = dict()
        partsToSizes = dict()
        key = uuid1()
        for w, futures in input_futures.items():
            self.rnks[w] = worker_info[w]["r"]
            parts = [(self.client.submit(
                CD._func_get_size_cl,
                future,
                workers=[w],
                key="%s-%s" % (key, idx)).result())
                for idx, future in enumerate(futures)]

            partsToSizes[worker_info[w]["r"]] = parts
            for p in parts:
                n_rows = n_rows + p

        key = uuid1()
        self.linear_models = [(w, self.client.submit(
            CD._func_create_model,
            comms.sessionId,
            **self.kwargs,
            workers=[w],
            key="%s-%s" % (key, idx)))
            for idx, w in enumerate(workers)]

        key = uuid1()
        linear_fit = dict([(worker_info[wf[0]]["r"], self.client.submit(
            CD._func_fit_colocated,
            wf[1],
            input_futures[wf[0]],
            n_rows, n_cols,
            partsToSizes,
            worker_info[wf[0]]["r"],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.linear_models)])

        wait(list(linear_fit.values()))
        raise_exception_from_futures(list(linear_fit.values()))

        comms.destroy()

        self.local_model = self.linear_models[0][1].result()
        self.coef_ = self.local_model.coef_
        self.intercept_ = self.local_model.intercept_

    def fit(self, X, y, force_colocality=False):
        """
        Fit the model with X and y. If force_colocality is set to True,
        the partitions of X and y will be re-distributed to force the
        co-locality.

        In some cases, data samples and their labels can be distributed
        into different workers by dask. In that case, force_colocality
        param can be set to True to re-arrange the data.

        Usually, you will not need to force co-locality if you pass the
        X and y as follows;

        fit(X["all_the_columns_but_labels"], X["labels"])

        You might want to force co-locality if you pass the X and y as
        follows;

        fit(X, y)

        because dask might have distributed the partitions of X and y
        into different workers.

        Parameters
        ----------
        X : dask cuDF dataframe (n_rows, n_features)
            Features for regression
        y : dask cuDF (n_rows, 1)
            Labels (outcome values)
        force_colocality: boolean (True: re-distributes the partitions
                          of X and y to force the co-locality of
                          the partitions)
        """

        if force_colocality:
            self._fit_with_colocality(X, y)
        else:
            self._fit(X, y)

    def predict(self, X):
        gpu_futures = self.client.sync(extract_ddf_partitions, X)

        worker_to_parts = OrderedDict()
        for w, p in gpu_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            worker_to_parts[w].append(p)

        key = uuid1()
        partsToRanks = [(self.rnks[wf[0]], self.client.submit(
            CD._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(gpu_futures)]

        N = X.shape[1]
        M = reduce(lambda a, b: a+b, map(lambda x: x[1], partsToRanks))

        key = uuid1()
        linear_pred = dict([(self.rnks[wf[0]], self.client.submit(
            CD._func_predict,
            wf[1],
            worker_to_parts[wf[0]],
            M, N,
            partsToRanks,
            self.rnks[wf[0]],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.linear_models)])

        wait(list(linear_pred.values()))
        raise_exception_from_futures(list(linear_pred.values()))

        out_futures = []
        completed_part_map = {}
        for rank, size in partsToRanks:
            if rank not in completed_part_map:
                completed_part_map[rank] = 0

            f = linear_pred[rank]
            out_futures.append(self.client.submit(
                CD._func_get_idx, f, completed_part_map[rank]))

            completed_part_map[rank] += 1

        return to_dask_cudf(out_futures)
        
    def get_param_names(self):
        return list(self.kwargs.keys())

