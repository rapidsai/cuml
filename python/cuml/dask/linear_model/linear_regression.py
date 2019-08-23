#
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

from cuml.dask.common.spmg_ipc import new_ipc_thread
from cuml.dask.common.utils import parse_host_port
from cuml.dask.common.utils import device_of_devicendarray
from cuml.dask.common.utils import build_host_dict

import cudf
import dask_cudf
import numpy as np

from dask import delayed
from dask.distributed import wait, default_client
from math import ceil
from numba import cuda
from toolz import first
from tornado import gen


class LinearRegression(object):
    """
    Model-Parallel Multi-GPU Linear Regression Model. Single Process Multi GPU
    supported currently
    """
    def __init__(self, fit_intercept=True, normalize=False):

        """
        Initializes the linear regression class.

        Parameters
        ----------
        fit_intercept: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self._model_fit = False
        self._consec_call = 0

    def _build_params_map(self):
        return {"fit_intercept": self.fit_intercept,
                "normalize": self.normalize}

    def fit(self, X, y):
        """
        Fits a multi-gpu linear regression model such that each of the
        resulting coefficients are also distributed across the GPUs.
        :param futures:
        :return:
        """
        client = default_client()

        self.dtype = X[X.columns[0]].compute().dtype

        coef, intercept, locations = client.sync(self._do_fit, X, y,
                                                 self.dtype)

        self.intercept = intercept
        self._locations = locations

        self._model_fit = True

        self._ncols = X.shape[1]

        self.coef_ = dask_cudf.from_delayed(coef)

    @gen.coroutine
    def _do_fit(self, X_df, y_df, dtype):

        client = default_client()

        # Finding location of parts of y_df to distribute columns of X_df
        loc_dict = {}
        yield wait(y_df)
        tt = yield client.who_has(y_df)
        location = tuple(tt.values())
        for i in range(X_df.npartitions):
            part_number = eval(list(tt.keys())[i])[1]
            loc_dict[part_number] = parse_host_port(str(location[i])[:-3])

        # Lets divide the columns evenly, matching the order of the labels
        part_size = ceil(X_df.shape[1] / X_df.npartitions)

        # We scatter delayed operations to gather columns on the workers
        scattered = []
        coefs = []
        for i in range(X_df.npartitions):
            up_limit = min((i+1)*part_size, X_df.shape[1])
            cols = X_df.columns.values[i*part_size:up_limit]
            loc_cudf = X_df[cols]
            yield wait(loc_cudf)
            scattered.append(client.submit(preprocess_on_worker,
                                           loc_cudf,
                                           workers=[loc_dict[i]]))
            yield wait(scattered)
            coefs.append(client.submit(dev_array_on_worker,
                                       up_limit - i*part_size,
                                       dtype=dtype,
                                       unique=np.random.randint(0, 1e6),
                                       workers=[loc_dict[i]]))
            yield wait(coefs)
            del(loc_cudf)

        # Break apart Dask.array/dataframe into chunks/parts
        # data_parts = map(delayed, scattered)
        data_parts = scattered
        label_parts = y_df.to_delayed()
        coef_parts = coefs

        # Arrange parts into pairs.  This enforces co-locality
        parts = list(map(delayed, zip(data_parts, label_parts, coef_parts)))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)

        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        who_has = yield client.who_has(parts)

        worker_parts = {}
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])

        """
        Create IP Handles on each worker hosting input data
        """

        # Format of input_devarrays = ([(X, y)..], dev)
        input_devarrays = [(worker, client.submit(fit_to_device_arrays,
                                                  part, workers=[worker]))
                           for worker, part in worker_parts.items()]

        yield wait(input_devarrays)

        """
        Gather IPC handles for each worker and call _fit() on each worker
        containing data.
        """

        # Last worker is the only one that can have less items.
        exec_node = loc_dict[X_df.npartitions-1]

        # Need to fetch parts on worker
        on_worker = list(filter(lambda x: x[0] == exec_node, input_devarrays))
        not_on_worker = list(filter(lambda x: x[0] != exec_node,
                                    input_devarrays))

        ipc_handles = [client.submit(get_input_ipc_handles, future,
                                     workers=[a_worker])
                       for a_worker, future in not_on_worker]

        raw_arrays = [future for a_worker, future in on_worker]

        # IPC Handles are loaded in separate threads on worker so they can be
        # used to make calls through cython
        # Calls _fit_on_worker defined in the bottom
        intercept = client.submit(_fit_on_worker, (ipc_handles, raw_arrays),
                                  self._build_params_map(),
                                  workers=[exec_node])

        yield wait(intercept)

        coef_series = [client.submit(coef_on_worker, coefs[i], i,
                                     X_df.shape[1],
                                     X_df.npartitions, loc_dict[i],
                                     workers=[loc_dict[i]])
                       for i in range(len(loc_dict))]

        # coef_on_worker(self, coef, locations, ncols, nparts, worker):

        raise gen.Return((coef_series, intercept, loc_dict))

    def predict(self, X):
        """
        Predict values for the multi-gpu linear regression model by making
        calls to the predict function with dask-cudf objects.

        :param df:
            a dask-cudf with data distributed one worker per GPU
        :return:
            a dask-cudf containing outputs of the linear regression
        """
        if self._model_fit:

            client = default_client()
            ret = client.sync(self._do_predict, X, self.coef_,
                              self._locations, self.intercept, self.dtype)

            ret = dask_cudf.from_delayed(ret)

            return ret

        else:
            raise ValueError('Model coefficients have not been fit. You need '
                             'to run the fit() method first. ')

    @gen.coroutine
    def _do_predict(self, X_df, coefs, loc_dict, intercept, dtype):
        client = default_client()

        part_size = ceil(X_df.shape[1] / X_df.npartitions)

        # We scatter delayed operations to gather columns on the workers
        scattered = []
        for i in range(X_df.npartitions):
            up_limit = min((i+1)*part_size, X_df.shape[1])
            cols = X_df.columns.values[i*part_size:up_limit]
            loc_cudf = X_df[cols]
            yield wait(loc_cudf)
            scattered.append(client.submit(preprocess_predict,
                                           loc_cudf,
                                           workers=[loc_dict[i]]))
            yield wait(scattered)
            del(loc_cudf)

        # Break apart Dask.array/dataframe into chunks/parts
        data_parts = scattered
        coef_parts = coefs.to_delayed()

        # Arrange parts into pairs.  This enforces co-locality
        parts = list(map(delayed, zip(data_parts, coef_parts)))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)

        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        who_has = yield client.who_has(parts)

        worker_parts = {}
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])

        """
        Create IP Handles on each worker hosting input data
        """

        # Format of input_devarrays = ([(X, y)..], dev)
        input_devarrays = [(worker, client.submit(predict_to_device_arrays,
                                                  part, worker, loc_dict,
                                                  X_df.npartitions,
                                                  dtype=dtype,
                                                  workers=[worker]))
                           for worker, part in worker_parts.items()]

        yield wait(input_devarrays)

        """
        Gather IPC handles for each worker and call _fit() on each worker
        containing data.
        """
        exec_node = loc_dict[X_df.npartitions-1]

        # Need to fetch parts on worker
        on_worker = list(filter(lambda x: x[0] == exec_node, input_devarrays))
        not_on_worker = list(filter(lambda x: x[0] != exec_node,
                                    input_devarrays))

        ipc_handles = [client.submit(get_input_ipc_handles, future,
                                     unique=np.random.randint(0, 1e6),
                                     workers=[a_worker])
                       for a_worker, future in not_on_worker]

        raw_arrays = [future for a_worker, future in on_worker]

        # IPC Handles are loaded in separate threads on worker so they can be
        # used to make calls through cython
        # Calls _predict_on_worker defined in the bottom
        ret = client.submit(_predict_on_worker, (ipc_handles, raw_arrays),
                            self.intercept, self._build_params_map(),
                            workers=[exec_node])

        yield wait(ret)

        dfs = [client.submit(series_on_worker, f, worker, loc_dict,
                             X_df.npartitions, X_df, workers=[worker])
               for worker, f in input_devarrays]

        return dfs

    def _build_host_dict(self, gpu_futures, client):

        who_has = client.who_has(gpu_futures)

        key_to_host_dict = {}
        for key in who_has:
            key_to_host_dict[key] = parse_host_port(who_has[key][0])

        hosts_to_key_dict = {}
        for key, host in key_to_host_dict.items():
            if host not in hosts_to_key_dict:
                hosts_to_key_dict[host] = set([key])
            else:
                hosts_to_key_dict[host].add(key)

        workers = [key[0] for key in list(who_has.values())]
        return build_host_dict(workers)


def _fit_on_worker(data, params):
    ipc_dev_list, devarrs_dev_list = data

    # Open 1 ipc thread per device
    open_ipcs = []
    for p, dev in ipc_dev_list:
        arrs = []
        for x, y, coef in p:
            arrs.append(x)
            arrs.append(y)
            arrs.append(coef)
        ipct = new_ipc_thread(arrs, dev)
        open_ipcs.append(ipct)

    alloc_info = []
    for t in open_ipcs:
        outsiders = t.info()
        triplet = []
        for i in range(0, len(outsiders), 3):
            triplet.append(outsiders[i])
            triplet.append(outsiders[i+1])
            triplet.append(outsiders[i+2])
            alloc_info.append(triplet)

    for p, dev in devarrs_dev_list:
        locals = []
        for X, coef, pred in p:
            locals.append(build_alloc_info(X)[0])
            locals.append(build_alloc_info(coef)[0])
            locals.append(build_alloc_info(pred)[0])
        alloc_info.append(locals)

    try:
        from cuml.linear_model.linear_regression_mg import LinearRegressionMG as cuOLS  # NOQA
        ols = cuOLS()
        intercept = ols._fit_mg(alloc_info, params)
    except ImportError:
        raise Exception("cuML has not been built with multiGPU support "
                        "enabled. Build with the --multigpu flag to enable"
                        " multiGPU support.")
    except Exception as e:
        print("Failure in Fit(): " + str(e))

    [t.close() for t in open_ipcs]
    # [t.join() for t in open_ipcs]

    return intercept


def _predict_on_worker(data, intercept, params):
    ipc_dev_list, devarrs_dev_list = data

    open_ipcs = []
    for p, dev in ipc_dev_list:
        arrs = []
        for mat, coef, pred in p:
            arrs.append(mat)
            arrs.append(coef)
            arrs.append(pred)
        ipct = new_ipc_thread(arrs, dev)
        open_ipcs.append(ipct)

    alloc_info = []
    for t in open_ipcs:
        outsiders = t.info()
        triplet = []
        for i in range(0, len(outsiders), 3):
            triplet.append(outsiders[i])
            triplet.append(outsiders[i+1])
            triplet.append(outsiders[i+2])
            alloc_info.append(triplet)

    for p, dev in devarrs_dev_list:
        locals = []
        for X, y, coef in p:
            locals.append(
                build_alloc_info(X, unique=np.random.randint(0, 1e6))[0])
            locals.append(
                build_alloc_info(y, unique=np.random.randint(0, 1e6))[0])
            locals.append(
                build_alloc_info(coef, unique=np.random.randint(0, 1e6))[0])
        alloc_info.append(locals)

    try:
        from cuml.linear_model.linear_regression_mg import LinearRegressionMG as cuOLS  # NOQA
        ols = cuOLS()
        ols._predict_mg(alloc_info, intercept, params)
    except ImportError:
        raise Exception("cuML has not been built with multiGPU support "
                        "enabled. Build with the --multigpu flag to enable"
                        " multiGPU support.")
    except Exception as e:
        print("Failure in predict(): " + str(e))

    [t.close() for t in open_ipcs]
    # [t.join() for t in open_ipcs]


def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)


def build_alloc_info(p, unique=0): return [p.__cuda_array_interface__]


def get_input_ipc_handles(arr, unique=0):

    arrs, dev = arr
    ret = [(X.get_ipc_handle(),
            y.get_ipc_handle(),
            coef.get_ipc_handle()) for X, y, coef in arrs]

    return ret, dev


def as_gpu_matrix(arr):
    blap = arr.compute()
    mat = blap.as_gpu_matrix(order="F")
    dev = device_of_devicendarray(mat)

    # Return canonical device id as string
    return mat, dev


def to_gpu_array(arr):

    mat = arr.to_gpu_array()
    dev = device_of_devicendarray(mat)

    # Return canonical device id as string
    return mat, dev


def fit_to_device_arrays(arr):
    """
    :param arr:
        A tuple in the form of (X, y, coef)
    :return:
    """

    mats = [(X.compute().as_gpu_matrix(order='F'),
             y.to_gpu_array(),
             coef) for X, y, coef in arr]

    dev = device_of_devicendarray(mats[0][0])

    # Return canonical device id as string
    return mats, dev


def predict_to_device_arrays(arr, worker, loc_dict, nparts, dtype):
    """
    :param arr:
        A tuple in the form of (X, y, coef)
    :return:
    """
    part_number = list(loc_dict.keys())[list(loc_dict.values()).index(worker)]

    mats = []
    for X, coef in arr:
        nrows = len(X)
        part_size = ceil(nrows / nparts)
        up_limit = min((part_number+1)*part_size, nrows)
        mat = X.compute().as_gpu_matrix(order='F')
        pred = cuda.to_device(np.zeros(up_limit-(part_number*part_size),
                                       dtype=dtype))
        mats.append([mat, coef.to_gpu_array(), pred])

    dev = device_of_devicendarray(mats[0][0])

    # Return canonical device id as string
    return mats, dev


def extract_part(data, part):
    return data[part]


def preprocess_on_worker(arr):
    return arr


def dev_array_on_worker(rows, dtype=np.float64, unique=0):
    return cuda.to_device(np.zeros(rows, dtype=dtype))


# Need to have different named function for predict to avoid
# dask key colision in case of same rows and columns between
# different arrays
def pred_array_on_worker(rows, cols, dtype=np.float64, unique=0):
    return cuda.to_device(np.zeros((rows, cols), dtype=dtype))


def preprocess_predict(arr):
    return arr


def series_on_worker(ary, worker, loc_dict, nparts, X):
    nrows = len(X)
    part_number = list(loc_dict.keys())[list(loc_dict.values()).index(worker)]
    part_size = ceil(nrows / nparts)
    up_limit = min((part_number+1)*part_size, nrows)
    if len(ary) == nparts - 1:
        idx = (part_number*len(ary[0][0][2]),
               (part_number+1)*len(ary[0][0][2]))
    else:
        idx = (up_limit-len(ary[0][0][2]), up_limit)

    ret = cudf.Series(ary[0][0][2], index=cudf.dataframe.RangeIndex(idx[0],
                                                                    idx[1]))
    return ret


def get_meta(df):
    ret = df.iloc[:0]
    return ret


def coef_on_worker(coef, part_number, ncols, nparts, worker):
    part_size = ceil(ncols / nparts)
    up_limit = min((part_number+1)*part_size, ncols)
    idx = (part_number*part_size, up_limit)
    ret = cudf.Series(coef, index=cudf.dataframe.RangeIndex(idx[0],
                                                            idx[1]))
    return ret
