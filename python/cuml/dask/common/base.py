from cuml.dask.common.input_utils import MGData
from cuml.dask.common.input_utils import to_output
from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.utils import MultiHolderLock

from dask_cudf.core import DataFrame as dcDataFrame

from uuid import uuid1
import dask

import cupy as cp
import numpy as np


class DelayedPredictionMixin(object):

    @staticmethod
    def _func_predict(f, df):

        res = [f.predict(d) for d in df]
        return res

    @staticmethod
    def _func_get_idx(f, idx):
        return f[idx]

    def predict(self, X, delayed=True, parallelism=25):
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

        if delayed:
            X_d = X.to_delayed()

            lock = dask.delayed(MultiHolderLock(parallelism),
                                pure=True)

            model = dask.delayed(self.local_model, pure=True)

            if isinstance(X, dcDataFrame):
                preds = [_delayed_predict(model, lock, part) for part in X_d]
                return dask.dataframe.from_delayed(preds)

            else:
                preds = [_delayed_predict(model, lock, part[0])
                         for part in X_d]

                # todo: add parameter for option of not checking directly
                dtype = X.dtype

                preds_arr = [
                    dask.array.from_delayed(pred,
                                            meta=cp.zeros(1, dtype=dtype),
                                            shape=(np.nan,),
                                            dtype=dtype)
                    for pred in preds]

                return dask.array.concatenate(preds_arr, axis=0,
                                              allow_unknown_chunksizes=True)

        else:
            X = X.persist()

            # todo: Push model as a future that will be shared per worker
            # rodo: push lock as a future that will be shared per worker

            data = MGData.single(X, client=self.client)

            key = uuid1()
            linear_pred = dict([(wf[0], self.client.submit(
                DelayedPredictionMixin._func_predict,
                self.local_model,
                data.worker_to_parts[wf[0]],
                key="%s-%s" % (key, idx),
                workers=[wf[0]]))
                for idx, wf in enumerate(data.gpu_futures)])

            raise_exception_from_futures(linear_pred.values())

            # loop to order the futures correctly to build the
            # dask-dataframe/array
            # todo: make this embarrassingly parallel
            results = []
            counters = dict.fromkeys(data.workers, 0)
            for idx, part in enumerate(data.gpu_futures):
                results.append(self.client.submit(
                    DelayedPredictionMixin._func_get_idx,
                    linear_pred[part[0]],
                    counters[part[0]])
                )
                counters[part[0]] = counters[part[0]] + 1

            return to_output(results, self.datatype)


@dask.delayed(pure=False, nout=1)
def _delayed_predict(model, lock, data):
    lock.acquire()
    ret = model.predict(data)
    lock.release()
    return ret 
