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

import dask
import math
import numpy as np

from cuml.dask.common.input_utils import DistributedDataHandler, \
    concatenate
from cuml.dask.common.utils import get_client, wait_and_raise_from_futures
from cuml.fil.fil import TreeliteModel


class BaseRandomForestModel(object):
    """
    BaseRandomForestModel defines functions used in both Random Forest
    Classifier and Regressor for Multi Node and Multi GPU models. The common
    functions are defined here and called from the main Random Forest Multi
    Node Multi GPU APIs. The functions defined here are not meant to be used
    as a part of the public API.
    """

    def _create_model(self, model_func,
                      client,
                      workers,
                      n_estimators,
                      base_seed,
                      **kwargs):

        self.client = get_client(client)
        self.workers = self.client.scheduler_info()['workers'].keys()
        self._set_internal_model(None)

        self.n_estimators_per_worker = \
            self._estimators_per_worker(n_estimators)
        if base_seed is None:
            base_seed = 0
        seeds = [base_seed]
        for i in range(1, len(self.n_estimators_per_worker)):
            sd = self.n_estimators_per_worker[i-1] + seeds[i-1]
            seeds.append(sd)

        self.rfs = {
            worker: self.client.submit(
                model_func,
                n_estimators=self.n_estimators_per_worker[n],
                seed=seeds[n],
                **kwargs,
                pure=False,
                workers=[worker],
            )
            for n, worker in enumerate(self.workers)
        }

        wait_and_raise_from_futures(list(self.rfs.values()))

    def _estimators_per_worker(self, n_estimators):
        n_workers = len(self.workers)
        if n_estimators < n_workers:
            raise ValueError(
                "n_estimators cannot be lower than number of dask workers."
            )

        n_est_per_worker = math.floor(n_estimators / n_workers)
        n_estimators_per_worker = \
            [n_est_per_worker for i in range(n_workers)]
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
        for i in range(remaining_est):
            n_estimators_per_worker[i] = (
                n_estimators_per_worker[i] + 1
            )
        return n_estimators_per_worker

    def _fit(self, model, dataset, convert_dtype):
        data = DistributedDataHandler.create(dataset, client=self.client)
        self.datatype = data.datatype
        if self.datatype == 'cudf':
            has_float64 = (dataset[0].dtypes.any() == np.float64)
        else:
            has_float64 = (dataset[0].dtype == np.float64)
        if has_float64:
            raise TypeError("To use Dask RF data should have dtype float32.")

        labels = self.client.persist(dataset[1])
        if self.datatype == 'cudf':
            self.num_classes = len(labels.unique())
        else:
            self.num_classes = \
                len(dask.array.unique(labels).compute())
        labels = self.client.persist(dataset[1])
        futures = list()
        for idx, (worker, worker_data) in \
                enumerate(data.worker_to_parts.items()):
            futures.append(
                self.client.submit(
                    _func_fit,
                    model[worker],
                    worker_data,
                    convert_dtype,
                    workers=[worker],
                    pure=False)
            )
        wait_and_raise_from_futures(futures)
        return self

    def _concat_treelite_models(self):
        """
        Convert the cuML Random Forest model present in different workers to
        the treelite format and then concatenate the different treelite models
        to create a single model. The concatenated model is then converted to
        bytes format.
        """
        model_serialized_futures = list()
        for w in self.workers:
            model_serialized_futures.append(
                dask.delayed(_get_serialized_model)
                (self.rfs[w]))
        mod_bytes = self.client.compute(model_serialized_futures, sync=True)
        last_worker = w
        model = self.rfs[last_worker].result()
        all_tl_mod_handles = [
                model._tl_handle_from_bytes(indiv_worker_model_bytes)
                for indiv_worker_model_bytes in mod_bytes
        ]

        model._concatenate_treelite_handle(all_tl_mod_handles)
        for tl_handle in all_tl_mod_handles:
            TreeliteModel.free_treelite_model(tl_handle)
        return model

    def _predict_using_fil(self, X, delayed, **kwargs):
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        data = DistributedDataHandler.create(X, client=self.client)
        self.datatype = data.datatype
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        return self._predict(X, delayed=delayed, **kwargs)

    def _get_params(self, deep):
        model_params = list()
        for idx, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_get_params,
                    self.rfs[worker],
                    deep,
                    workers=[worker]
                )
            )
        params_of_each_model = self.client.gather(model_params, errors="raise")
        return params_of_each_model

    def _set_params(self, **params):
        model_params = list()
        for idx, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_set_params,
                    self.rfs[worker],
                    **params,
                    workers=[worker]
                )
            )
        wait_and_raise_from_futures(model_params)
        return self

    def _print_summary(self):
        """
        Print the summary of the forest used to train and test the model.
        """
        futures = list()
        for n, w in enumerate(self.workers):
            futures.append(
                self.client.submit(
                    _print_summary_func,
                    self.rfs[w],
                    workers=[w],
                )
            )

            wait_and_raise_from_futures(futures)
        return self

    def _print_detailed(self):
        """
        Print the summary of the forest used to train and test the model.
        """
        futures = list()
        for n, w in enumerate(self.workers):
            futures.append(
                self.client.submit(
                    _print_detailed_func,
                    self.rfs[w],
                    workers=[w],
                )
            )
            wait_and_raise_from_futures(futures)
        return self


def _func_fit(model, input_data, convert_dtype):
    X = concatenate([item[0] for item in input_data])
    y = concatenate([item[1] for item in input_data])
    return model.fit(X, y, convert_dtype)


def _print_summary_func(model):
    model.print_summary()


def _print_detailed_func(model):
    model.print_detailed()


def _func_get_params(model, deep):
    return model.get_params(deep)


def _func_set_params(model, **params):
    return model.set_params(**params)


def _get_serialized_model(model):
    return model._get_serialized_model()
