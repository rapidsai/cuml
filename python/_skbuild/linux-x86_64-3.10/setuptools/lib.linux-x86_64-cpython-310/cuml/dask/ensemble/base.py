# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from cuml.fil.fil import TreeliteModel
from cuml.dask.common.utils import get_client, wait_and_raise_from_futures
from cuml.dask.common.input_utils import DistributedDataHandler, concatenate
from dask.distributed import Future
from collections.abc import Iterable
from cuml import using_output_type
import warnings
from cuml.internals.safe_imports import gpu_only_import
import dask
import json
import math
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


class BaseRandomForestModel(object):
    """
    BaseRandomForestModel defines functions used in both Random Forest
    Classifier and Regressor for Multi Node and Multi GPU models. The common
    functions are defined here and called from the main Random Forest Multi
    Node Multi GPU APIs. The functions defined here are not meant to be used
    as a part of the public API.
    """

    def _create_model(
        self,
        model_func,
        client,
        workers,
        n_estimators,
        base_seed,
        ignore_empty_partitions,
        **kwargs,
    ):

        self.client = get_client(client)
        if workers is None:
            # Default to all workers
            workers = list(self.client.scheduler_info()["workers"].keys())
        self.workers = workers
        self._set_internal_model(None)
        self.active_workers = list()
        self.ignore_empty_partitions = ignore_empty_partitions
        self.n_estimators = n_estimators

        self.n_estimators_per_worker = self._estimators_per_worker(
            n_estimators
        )
        if base_seed is None:
            base_seed = 0
        seeds = [base_seed]
        for i in range(1, len(self.n_estimators_per_worker)):
            sd = self.n_estimators_per_worker[i - 1] + seeds[i - 1]
            seeds.append(sd)

        self.rfs = {
            worker: self.client.submit(
                model_func,
                n_estimators=self.n_estimators_per_worker[n],
                random_state=seeds[n],
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
        n_estimators_per_worker = [n_est_per_worker for i in range(n_workers)]
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
        for i in range(remaining_est):
            n_estimators_per_worker[i] = n_estimators_per_worker[i] + 1
        return n_estimators_per_worker

    def _fit(self, model, dataset, convert_dtype, broadcast_data):
        data = DistributedDataHandler.create(dataset, client=self.client)
        self.active_workers = data.workers
        self.datatype = data.datatype
        if self.datatype == "cudf":
            has_float64 = (dataset[0].dtypes == np.float64).any()
        else:
            has_float64 = dataset[0].dtype == np.float64
        if has_float64:
            raise TypeError("To use Dask RF data should have dtype float32.")

        labels = self.client.persist(dataset[1])
        if self.datatype == "cudf":
            self.num_classes = len(labels.unique())
        else:
            self.num_classes = len(dask.array.unique(labels).compute())

        combined_data = (
            list(map(lambda x: x[1], data.gpu_futures))
            if broadcast_data
            else None
        )

        futures = list()
        for idx, (worker, worker_data) in enumerate(
            data.worker_to_parts.items()
        ):
            futures.append(
                self.client.submit(
                    _func_fit,
                    model[worker],
                    combined_data if broadcast_data else worker_data,
                    convert_dtype,
                    workers=[worker],
                    pure=False,
                )
            )

        self.n_active_estimators_per_worker = []
        for worker in data.worker_to_parts.keys():
            n = self.workers.index(worker)
            n_est = self.n_estimators_per_worker[n]
            self.n_active_estimators_per_worker.append(n_est)

        if len(self.workers) > len(self.active_workers):
            if self.ignore_empty_partitions:
                curent_estimators = (
                    self.n_estimators
                    / len(self.workers)
                    * len(self.active_workers)
                )
                warn_text = (
                    f"Data was not split among all workers "
                    f"using only {self.active_workers} workers to fit."
                    f"This will only train {curent_estimators}"
                    f" estimators instead of the requested "
                    f"{self.n_estimators}"
                )
                warnings.warn(warn_text)
            else:
                raise ValueError(
                    "Data was not split among all workers. "
                    "Re-run the code or "
                    "use ignore_empty_partitions=True"
                    " while creating model"
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
        for w in self.active_workers:
            model_serialized_futures.append(
                dask.delayed(_get_serialized_model)(self.rfs[w])
            )
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

    def _partial_inference(self, X, op_type, delayed, **kwargs):
        data = DistributedDataHandler.create(X, client=self.client)
        combined_data = list(map(lambda x: x[1], data.gpu_futures))

        func = (
            _func_predict_partial
            if op_type == "regression"
            else _func_predict_proba_partial
        )

        partial_infs = list()
        for worker in self.active_workers:
            partial_infs.append(
                self.client.submit(
                    func,
                    self.rfs[worker],
                    combined_data,
                    **kwargs,
                    workers=[worker],
                    pure=False,
                )
            )
        partial_infs = dask.delayed(dask.array.concatenate)(
            partial_infs, axis=1, allow_unknown_chunksizes=True
        )
        return partial_infs

    def _predict_using_fil(self, X, delayed, **kwargs):
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        data = DistributedDataHandler.create(X, client=self.client)
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        return self._predict(
            X, delayed=delayed, output_collection_type=data.datatype, **kwargs
        )

    def _get_params(self, deep):
        model_params = list()
        for idx, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_get_params, self.rfs[worker], deep, workers=[worker]
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
                    workers=[worker],
                )
            )
        wait_and_raise_from_futures(model_params)
        return self

    def _get_summary_text(self):
        """
        Obtain the summary of the forest as text
        """
        futures = list()
        for n, w in enumerate(self.workers):
            futures.append(
                self.client.submit(
                    _get_summary_text_func,
                    self.rfs[w],
                    workers=[w],
                )
            )
        all_dump = self.client.gather(futures, errors="raise")
        return "\n".join(all_dump)

    def _get_detailed_text(self):
        """
        Obtain the detailed information of the forest as text
        """
        futures = list()
        for n, w in enumerate(self.workers):
            futures.append(
                self.client.submit(
                    _get_detailed_text_func,
                    self.rfs[w],
                    workers=[w],
                )
            )
        all_dump = self.client.gather(futures, errors="raise")
        return "\n".join(all_dump)

    def _get_json(self):
        """
        Export the Random Forest model as a JSON string
        """
        dump = list()
        for n, w in enumerate(self.workers):
            dump.append(
                self.client.submit(
                    _get_json_func,
                    self.rfs[w],
                    workers=[w],
                )
            )
        all_dump = self.client.gather(dump, errors="raise")
        combined_dump = []
        for e in all_dump:
            obj = json.loads(e)
            combined_dump.extend(obj)
        return json.dumps(combined_dump)

    def get_combined_model(self):
        """
        Return single-GPU model for serialization.

        Returns
        -------

        model : Trained single-GPU model or None if the model has not
                yet been trained.
        """

        # set internal model if it hasn't been accessed before
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())

        internal_model = self._check_internal_model(self._get_internal_model())

        if isinstance(self.internal_model, Iterable):
            # This function needs to return a single instance of cuml.Base,
            # even if the class is just a composite.
            raise ValueError(
                "Expected a single instance of cuml.Base "
                "but got %s instead." % type(self.internal_model)
            )

        elif isinstance(self.internal_model, Future):
            internal_model = self.internal_model.result()

        return internal_model

    def apply_reduction(self, reduce, partial_infs, datatype, delayed):
        """
        Reduces the partial inferences to obtain the final result. The workers
        didn't have the same number of trees to form their predictions. To
        correct for this worker's predictions are weighted differently during
        reduction.
        """
        workers_weights = np.array(self.n_active_estimators_per_worker)
        workers_weights = workers_weights[workers_weights != 0]
        workers_weights = workers_weights / workers_weights.sum()
        workers_weights = cp.array(workers_weights)
        unique_classes = (
            None
            if not hasattr(self, "unique_classes")
            else self.unique_classes
        )
        delayed_local_array = dask.delayed(reduce)(
            partial_infs, workers_weights, unique_classes
        )
        delayed_res = dask.array.from_delayed(
            delayed_local_array, shape=(np.nan, np.nan), dtype=np.float32
        )
        if delayed:
            return delayed_res
        else:
            return delayed_res.persist()


def _func_fit(model, input_data, convert_dtype):
    X = concatenate([item[0] for item in input_data])
    y = concatenate([item[1] for item in input_data])
    return model.fit(X, y, convert_dtype)


def _func_predict_partial(model, input_data, **kwargs):
    """
    Whole dataset inference with part of the model (trees at disposal locally).
    Transfer dataset instead of model. Interesting when model is larger
    than dataset.
    """
    X = concatenate(input_data)
    with using_output_type("cupy"):
        prediction = model.predict(X, **kwargs)
        return cp.expand_dims(prediction, axis=1)


def _func_predict_proba_partial(model, input_data, **kwargs):
    """
    Whole dataset inference with part of the model (trees at disposal locally).
    Transfer dataset instead of model. Interesting when model is larger
    than dataset.
    """
    X = concatenate(input_data)
    with using_output_type("cupy"):
        prediction = model.predict_proba(X, **kwargs)
        return cp.expand_dims(prediction, axis=1)


def _get_summary_text_func(model):
    return model.get_summary_text()


def _get_detailed_text_func(model):
    return model.get_detailed_text()


def _get_json_func(model):
    return model.get_json()


def _func_get_params(model, deep):
    return model.get_params(deep)


def _func_set_params(model, **params):
    return model.set_params(**params)


def _get_serialized_model(model):
    return model._get_serialized_model()
