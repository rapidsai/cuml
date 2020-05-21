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

import cudf.comm.serialize  # noqa: F401
import cupy as cp
import dask
import numpy as np

from cuml import Base
from cuml.common.array import CumlArray
import cuml.common.logger as logger
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.dask.common.comms import CommsContext
from cuml.dask.common.input_utils import DistributedDataHandler

from dask_cudf.core import DataFrame as dcDataFrame
from dask.distributed import default_client
from functools import wraps
from toolz import first


class BaseEstimator(object):

    def __init__(self, client=None, verbosity=logger.LEVEL_INFO, **kwargs):
        """
        Constructor for distributed estimators
        """
        self.client = default_client() if client is None else client
        self.verbosity = verbosity
        self.kwargs = kwargs

    @staticmethod
    @dask.delayed
    def _get_model_attr(model, name):
        if hasattr(model, name):
            return getattr(model, name)
        else:
            raise ValueError("Attribute %s does not exist on model %s" %
                             (name, type(model)))

    def __getattr__(self, attr):
        """
        Method gives access to the correct format of cuml Array attribute to
        the users and proxies attributes to the underlying trained model.

        If the attribute being requested is not directly on the local object,
        this function will see if the local object contains the attribute
        prefixed with an _. In the case the attribute does not exist on this
        local instance, the request will be proxied to self.local_model and
        will be fetched either locally or remotely depending on whether
        self.local_model is a local object instance or a future.
        """
        real_name = '_' + attr

        # First check locally for attr
        if attr in self.__dict__:
            ret_attr = self.__dict__[attr]

        # Next check locally for _ prefixed attr
        elif real_name in self.__dict__:
            ret_attr = self.__dict__[real_name]

        # Finally, check the trained model (this is done as a
        # last resort since fetching the attribute from the
        # distributed model will incur a higher cost than
        # local attributes.
        elif "local_model" in self.__dict__:
            local_model = self.__dict__["local_model"]

            if isinstance(local_model, Base):
                # If model is not distributed, just return the
                # requested attribute
                ret_attr = getattr(local_model, attr)
            else:
                # Otherwise, fetch the attribute from the distributed
                # model and return it
                ret_attr = BaseEstimator._get_model_attr(
                    self.__dict__["local_model"], attr).compute()
        else:
            raise ValueError("Attribute %s not found in %s" %
                             (attr, type(self)))

        if isinstance(ret_attr, CumlArray):
            return ret_attr.to_output(self.output_type)
        else:
            return ret_attr


class DelayedParallelFunc(object):
    def _run_parallel_func(self,
                           func,
                           X,
                           n_dims=1,
                           delayed=True,
                           output_futures=False,
                           output_dtype=None,
                           output_collection_type=None,
                           **kwargs):
        """
        Runs a function embarrassingly parallel on a set of workers while
        reusing instances of models and constraining the number of
        tasks that can execute concurrently on each worker.

        Note that this mixin assumes the subclass has been trained and
        includes a `self.local_model` attribute containing a subclass
        of `cuml.Base`.

        This is intended to abstract functions like predict, transform, and
        score, which can execute embarrassingly parallel but need addition
        execution constraints which result from the more limited GPU
        resources.

        Parameters
        ----------
        func : dask.delayed function to propagate to the workers to execute
               embarrassingly parallel, shared between tasks on each worker

        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return a lazy (delayed) object?

        output_futures : bool returns the futures pointing the to the resuls
                         of the parallel function executions on the workers,
                         rather than a dask collection object.

        output_collection_type : None or a string in {'cupy', 'cudf'}
            Choose to return the resulting collection as a CuPy backed
            dask.array or a dask_cudf.DataFrame. If None, will use the same
            collection type as used in the input of fit.
            Unused if output_futures=True.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """
        if output_collection_type is None:
            output_collection_type = self.datatype

        X_d = X.to_delayed()

        model = dask.delayed(self.local_model, pure=True, traverse=False)

        func = dask.delayed(func, pure=False, nout=1)

        if isinstance(X, dcDataFrame):

            preds = [func(model, part, **kwargs) for part in X_d]
            dtype = first(X.dtypes) if output_dtype is None else output_dtype

        else:
            preds = [func(model, part[0])
                     for part in X_d]
            dtype = X.dtype if output_dtype is None else output_dtype

        # TODO: Put the following conditionals in a
        #  `to_delayed_output()` function
        # TODO: Add eager path back in
        if output_collection_type == 'cupy':

            # todo: add parameter for option of not checking directly

            shape = (np.nan,) * n_dims
            preds_arr = [
                dask.array.from_delayed(pred,
                                        meta=cp.zeros(1, dtype=dtype),
                                        shape=shape,
                                        dtype=dtype)
                for pred in preds]

            if output_futures:
                return self.client.compute(preds)
            else:
                output = dask.array.concatenate(preds_arr, axis=0,
                                                allow_unknown_chunksizes=True
                                                )

                return output if delayed else output.persist()
        else:
            if output_futures:
                return self.client.compute(preds)
            else:
                output = dask.dataframe.from_delayed(preds)
                return output if delayed else output.persist()


class DelayedPredictionProbaMixin(DelayedParallelFunc):

    def _predict_proba(self, X, delayed=True, **kwargs):
        return self._run_parallel_func(func=_predict_proba_func, X=X,
                                       n_dims=2, delayed=delayed, **kwargs)


class DelayedPredictionMixin(DelayedParallelFunc):

    def _predict(self, X, delayed=True, **kwargs):
        return self._run_parallel_func(func=_predict_func, X=X,
                                       n_dims=1, delayed=delayed,
                                       **kwargs)


class DelayedTransformMixin(DelayedParallelFunc):

    def _transform(self, X, n_dims=1, delayed=True, **kwargs):
        return self._run_parallel_func(func=_transform_func,
                                       X=X,
                                       n_dims=n_dims,
                                       delayed=delayed,
                                       **kwargs)


class DelayedInverseTransformMixin(DelayedParallelFunc):

    def _inverse_transform(self, X, n_dims=1, delayed=True, **kwargs):
        return self._run_parallel_func(func=_inverse_transform_func,
                                       X=X,
                                       n_dims=n_dims,
                                       delayed=delayed,
                                       **kwargs)


class SyncFitMixinLinearModel(object):

    def _fit(self, model_func, data):

        n_cols = data[0].shape[1]

        data = DistributedDataHandler.create(data=data, client=self.client)
        self.datatype = data.datatype

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)
        self.ranks = data.ranks

        lin_models = dict([(data.worker_info[worker_data[0]]["rank"],
                            self.client.submit(
            model_func,
            comms.sessionId,
            self.datatype,
            **self.kwargs,
            pure=False,
            workers=[worker_data[0]]))

            for worker, worker_data in
            enumerate(data.worker_to_parts.items())])

        lin_fit = dict([(worker_data[0], self.client.submit(
            _func_fit,
            lin_models[data.worker_info[worker_data[0]]["rank"]],
            worker_data[1],
            data.total_rows,
            n_cols,
            data.parts_to_sizes[data.worker_info[worker_data[0]]["rank"]],
            data.worker_info[worker_data[0]]["rank"],
            pure=False,
            workers=[worker_data[0]]))

            for worker, worker_data in
            enumerate(data.worker_to_parts.items())])

        wait_and_raise_from_futures(list(lin_fit.values()))

        comms.destroy()
        return lin_models


def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
    return f.fit(data, n_rows, n_cols, partsToSizes, rank)


def mnmg_import(func):

    @wraps(func)
    def check_cuml_mnmg(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError:
            raise RuntimeError("cuML has not been built with multiGPU support "
                               "enabled. Build with the --multigpu flag to"
                               " enable multiGPU support.")

    return check_cuml_mnmg


def _predict_func(model, data, **kwargs):
    return model.predict(data, **kwargs)


def _predict_proba_func(model, data, **kwargs):
    return model.predict_proba(data, **kwargs)


def _transform_func(model, data, **kwargs):
    return model.transform(data, **kwargs)


def _inverse_transform_func(model, data, **kwargs):
    return model.inverse_transform(data, **kwargs)
