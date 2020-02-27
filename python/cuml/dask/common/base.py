from cuml.dask.common.input_utils import MGData
from cuml.dask.common.input_utils import to_output
from cuml.dask.common.utils import MultiHolderLock

from dask_cudf.core import DataFrame as dcDataFrame

from uuid import uuid1
import dask

import cupy as cp
import numpy as np


class DelayedParallelFunc(object):
    def _run_parallel_func(self,
                           func,
                           X,
                           delayed=True,
                           parallelism=25,
                           output_futures=False):
        """
        Make predictions for X and returns a y_pred.

        Parameters
        ----------
        func : dask.delayed function to propagate to the workers to execute
               embarrassingly parallel, shared between tasks on each worker
               and constrained by a holder lock
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        if delayed:
            X_d = X.to_delayed()

            lock = dask.delayed(MultiHolderLock(parallelism),
                                pure=True)

            model = dask.delayed(self.local_model, pure=True)

            func = dask.delayed(func, pure=False, nout=1)

            if isinstance(X, dcDataFrame):
                preds = [func(model, lock, part) for part in X_d]
                return preds if output_futures \
                    else dask.dataframe.from_delayed(preds)

            else:
                preds = [func(model, lock, part[0])
                         for part in X_d]

                # todo: add parameter for option of not checking directly
                dtype = X.dtype

                preds_arr = [
                    dask.array.from_delayed(pred,
                                            meta=cp.zeros(1, dtype=dtype),
                                            shape=(np.nan,),
                                            dtype=dtype)
                    for pred in preds]

                if output_futures:
                    return preds_arr
                else:
                    return dask.array.concatenate(preds_arr, axis=0,
                                                  allow_unknown_chunksizes=True
                                                  )

        else:
            X = X.persist()

            data = MGData.single(X, client=self.client)
            scattered = self.client.scatter(self.local_model,
                                            workers=data.workers,
                                            broadcast=True,
                                            hash=False)

            lock = self.client.scatter(MultiHolderLock(parallelism),
                                       workers=data.workers,
                                       broadcast=True,
                                       hash=False)

            key = uuid1()
            func_futures = [self.client.submit(
                func,
                scattered,
                lock,
                p,
                key=key) for w, p in data.gpu_futures]

            return func_futures if output_futures \
                else to_output(func_futures, self.datatype)


class DelayedPredictionMixin(DelayedParallelFunc):

    def _predict(self, X, delayed=True, parallelism=25):
        """
        Make predictions for X and returns a y_pred.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        return self._run_parallel_func(_predict_func, X, delayed, parallelism)


class DelayedTransformMixin(DelayedParallelFunc):

    def _transform(self, X, delayed=True, parallelism=25):
        """
        Make predictions for X and returns a y_pred.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        parallelism : int
            Amount of concurrent partitions that will be processed
            per worker. This bounds the total amount of temporary
            workspace memory on the GPU that will need to be allocated
            at any time.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """

        return self._run_parallel_func(_transform_func, X, delayed, parallelism)


def _predict_func(model, lock, data):
    lock.acquire()
    ret = model.predict(data)
    lock.release()
    return ret 


def _transform_func(model, lock, data):
    lock.acquire()
    ret = model.transform(data)
    lock.release()
    return ret
