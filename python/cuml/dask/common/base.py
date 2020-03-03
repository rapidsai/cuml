from cuml.dask.common.input_utils import MGData
from cuml.dask.common.input_utils import to_output
from cuml.dask.common.utils import MultiHolderLock

from dask.distributed import default_client

from dask_cudf.core import DataFrame as dcDataFrame

from uuid import uuid1
import dask

from toolz import first

import cupy as cp
import numpy as np

from cuml.dask.common.utils import patch_cupy_sparse_serialization


class BaseEstimator(object):

    def __init__(self, client=None, **kwargs):
        """
        Constructor for distributed estimators
        """
        self.client = default_client() if client is None else client

        patch_cupy_sparse_serialization(self.client)

        self.kwargs = kwargs


class DelayedParallelFunc(object):
    def _run_parallel_func(self,
                           func,
                           X,
                           n_dims=1,
                           delayed=True,
                           parallelism=5,
                           output_futures=False,
                           output_dtype=None):
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
               and constrained by a holder lock

        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return a lazy (delayed) object?

        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        output_futures : bool returns the futures pointing the to the resuls
                         of the parallel function executions on the workers,
                         rather than a dask collection object.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        X_d = X.to_delayed()

        lock = dask.delayed(MultiHolderLock(parallelism),
                            pure=True)

        model = dask.delayed(self.local_model, pure=True)

        func = dask.delayed(func, pure=False, nout=1)

        if isinstance(X, dcDataFrame):

            preds = [func(model, lock, part) for part in X_d]
            dtype = first(X.dtypes) if output_dtype is None else output_dtype

        else:
            preds = [func(model, lock, part[0])
                     for part in X_d]
            dtype = X.dtype if output_dtype is None else output_dtype

        # TODO: Put the following conditionals in a `to_delayed_output()` function
        if self.datatype == 'cupy':

            # todo: add parameter for option of not checking directly

            shape = tuple([np.nan for dim in range(n_dims)])
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
            output = preds if output_futures \
                else dask.dataframe.from_delayed(preds)

            return output if delayed else output.persist()


class DelayedPredictionMixin(DelayedParallelFunc):

    def _predict(self, X, delayed=True, parallelism=5):
        """
        Make predictions for X and returns a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return lazy (delayed) result?

        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        return self._run_parallel_func(_predict_func, X, 1, delayed, parallelism)


class DelayedTransformMixin(DelayedParallelFunc):

    def _transform(self, X, n_dims=1, delayed=True, parallelism=25):
        """
        Call transform on the partitions of X and produce a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return lazy (delayed) result?

        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        return self._run_parallel_func(_transform_func,
                                       X,
                                       n_dims,
                                       delayed,
                                       parallelism)


class DelayedInverseTransformMixin(DelayedParallelFunc):

    def _transform(self, X, n_dims=1, delayed=True, parallelism=25):
        """
        Call transform on the partitions of X and produce a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return lazy (delayed) result?

        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        return self._run_parallel_func(_inverse_transform_func,
                                       X,
                                       n_dims,
                                       delayed,
                                       parallelism)


def _predict_func(model, lock, data):
    lock.acquire()
    print(str(data))
    ret = model.predict(data)
    lock.release()
    return ret


def _transform_func(model, lock, data):
    lock.acquire()
    ret = model.transform(data)
    lock.release()
    return ret


def _inverse_transform_func(model, lock, data):
    lock.acquire()
    ret = model.inverse_transform(data)
    lock.release()
    return ret
