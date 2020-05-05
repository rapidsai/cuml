import math

from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.input_utils import DistributedDataHandler, \
    concatenate

from dask.distributed import wait

from uuid import uuid1


class BaseRandomForestModel(object):

    def _create_the_model(self, model_func, **kwargs):
        n_workers = len(self.workers)
        if self.n_estimators < n_workers:
            raise ValueError(
                "n_estimators cannot be lower than number of dask workers."
            )

        n_est_per_worker = math.floor(self.n_estimators / n_workers)

        for i in range(n_workers):
            self.n_estimators_per_worker.append(n_est_per_worker)

        remaining_est = self.n_estimators - (n_est_per_worker * n_workers)

        for i in range(remaining_est):
            self.n_estimators_per_worker[i] = (
                self.n_estimators_per_worker[i] + 1
            )
        seeds = list()
        seeds.append(0)
        for i in range(1, len(self.n_estimators_per_worker)):
            sd = self.n_estimators_per_worker[i-1] + seeds[i-1]
            seeds.append(sd)

        key = str(uuid1())
        self.rfs = {
            worker: self.client.submit(
                model_func,
                n_estimators=self.n_estimators_per_worker[n],
                seed=seeds[n],
                **kwargs,
                key="%s-%s" % (key, n),
                workers=[worker],
            )
            for n, worker in enumerate(self.workers)
        }

        rfs_wait = list()
        for r in self.rfs.values():
            rfs_wait.append(r)

        wait(rfs_wait)
        raise_exception_from_futures(rfs_wait)

    def _fit(self, model, dataset, convert_dtype):
        data = DistributedDataHandler.create(dataset, client=self.client)
        self.datatype = data.datatype
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
        wait(futures)
        raise_exception_from_futures(futures)
        return self

    def _concat_treelite_models(self):
        """
        Convert the cuML Random Forest model present in different workers to
        the treelite format and then concatenate the different treelite models
        to create a single model. The concatenated model is then converted to
        bytes format.
        """
        mod_bytes = []
        for w in self.workers:
            mod_bytes.append(self.rfs[w].result().model_pbuf_bytes)
        last_worker = w
        all_tl_mod_handles = []
        model = self.rfs[last_worker].result()
        for n in range(len(self.workers)):
            all_tl_mod_handles.append(model._tl_model_handles(mod_bytes[n]))
        concat_model_handle = model._concatenate_treelite_handle(
            treelite_handle=all_tl_mod_handles)

        model._concatenate_model_bytes(concat_model_handle)
        return model

    def _predict_using_fil(self, X, delayed, **kwargs):
        data = DistributedDataHandler.create(X, client=self.client)
        self.datatype = data.datatype
        return self._predict(X, delayed=delayed, **kwargs)

    def _get_params(self, deep):
        model_params = list()
        for n, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_get_params,
                    self.rfs[worker],
                    deep,
                    workers=[worker]
                )
            )
        wait(model_params)
        raise_exception_from_futures(model_params)
        params_of_each_model = list()
        for i in range(len(model_params)):
            params_of_each_model.append(model_params[i].result())
        return params_of_each_model

    def _set_params(self, worker_numb, **params):
        model_params = list()
        workers = [worker for worker in self.workers]
        for i in worker_numb:
            model_params.append(
                self.client.submit(
                    _func_set_params,
                    self.rfs[workers[i]],
                    **params,
                    workers=[workers[i]]
                )
            )
        wait(model_params)
        raise_exception_from_futures(model_params)
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

        wait(futures)
        raise_exception_from_futures(futures)
        return self


def _func_fit(model, input_data, convert_dtype):
    X = concatenate([item[0] for item in input_data])
    y = concatenate([item[1] for item in input_data])
    return model.fit(X, y, convert_dtype)


def _print_summary_func(model):
    model.print_summary()


def _func_get_params(model, deep):
    return model.get_params(deep)


def _func_set_params(model, **params):
    return model.set_params(**params)
