from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, futures_of
import dask.array as da
from cuml.dask.linear_model import LinearRegression
from cuml.dask.datasets.regression import make_regression
from cuml.dask.common.comms import CommsContext
import cudf
import dask_cudf
import numpy as np
import sys
from time import time, sleep
import warnings
import rmm
import cupy as cp
import dask
import numpy as np
from cuml.dask.common.dask_arr_utils import extract_arr_partitions

base_n_points = 250_000_000
point_multiplier = np.asarray([10])
base_n_features = np.asarray([250])

ideal_benchmark_f = open('ideal_benchmark_f.csv', 'a')

def _mse(ytest, yhat):
    if ytest.shape == yhat.shape:
        return (cp.mean((ytest - yhat) ** 2), ytest.shape[0])
    else:
        print("sorry")


def dask_mse(ytest, yhat, client, workers):
    ytest_parts = client.sync(extract_arr_partitions, ytest, client)
    yhat_parts = client.sync(extract_arr_partitions, yhat, client)
    mse_parts = np.asarray([client.submit(_mse, ytest_parts[i][1], yhat_parts[i][1]).result() for i in range(len(ytest_parts))])
    mse_parts[:, 0] = mse_parts[:, 0] * mse_parts[:, 1]
    return np.sum(mse_parts[:, 0]) / np.sum(mse_parts[:, 1])


def set_alloc():
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def make_client(n_workers=2):
    cluster = LocalCUDACluster(n_workers=n_workers)
    client = Client(cluster)
    return client


def check_order(x):
    print(x.flags.f_contiguous, x.strides)
    return x


def transpose_and_move(X, client, workers, n_samples, n_workers, n_features):
    futures = client.sync(extract_arr_partitions, X, client)
    futures = [client.submit(cp.array, futures[i][1], order="F", dtype=cp.float32, workers=[workers[i]]) for i in range(len(futures))]
    wait([futures])

    X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32), shape=(n_samples / n_workers, n_features), dtype=cp.float32) for x in futures]
    X = da.concatenate(X, axis=0, allow_unknown_chunksizes=True)
    # X_arr = X_arr.map_blocks(check_order, dtype=cp.float32)
    return X
    # return X_arr


def run_ideal_benchmark(n_workers=2):

    for n_points in base_n_points * point_multiplier:
        for n_features in base_n_features:
            fit_time = np.zeros(5)
            pred_time = np.zeros(5)
            for i in range(1):
                try:
                    cluster = LocalCUDACluster(n_workers=n_workers)
                    client = Client(cluster)
                    client.run(set_alloc)

                    workers = list(client.has_what().keys())
                    print(workers)

                    n_samples = n_points / n_features
                    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features / 10, n_parts=n_workers)

                    X = X.rechunk((n_samples / n_workers, n_features))
                    y = y.rechunk(n_samples / n_workers )

                    # Transpose X
                    print("Before transpose X")
                    X = transpose_and_move(X, client, workers, n_samples, n_workers, n_features)
                    print("After transpose X")
                    print(client.has_what())
                    # X.map_blocks(check_order, dtype=cp.float32)

                    # Transpose Y
                    print("Before transpose y")
                    y = transpose_and_move(y, client, workers, n_samples, n_workers, n_features)
                    print("After transpose y")
                    print(client.has_what())
                    # y.map_blocks(check_order, dtype=cp.float32)
                    
                    lr = LinearRegression(client=client)

                    start_fit_time = time()
                    print("Before Fit")
                    lr.fit(X, y)
                    print("After Fit")
                    end_fit_time = time()
                    print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Fit Time: ", end_fit_time - start_fit_time)
                    fit_time[i] = end_fit_time - start_fit_time

                    start_pred_time = time()
                    preds = lr.predict(X)
                    parts = client.sync(extract_arr_partitions, preds, client)
                    wait([p for w, p in parts])
                    print(parts)
                    # wait(client.compute(preds))
                    end_pred_time = time()
                    print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Predict Time: ", end_pred_time - start_pred_time)
                    pred_time[i] = end_pred_time - start_pred_time

                    mse = dask_mse(y, preds, client, workers)
                    print(mse)

                    del X, y, preds

                except Exception as e:
                    print(e)
                    continue

                finally:
                    cluster.close()
                    client.close()

            fit_stats = [np.mean(fit_time), np.min(fit_time), np.var(fit_time)]
            pred_stats = [np.mean(pred_time), np.min(pred_time), np.var(pred_time)]
            ideal_benchmark_f.write(','.join(map(str, [n_workers, n_samples, n_features] + fit_stats + pred_stats)))
            ideal_benchmark_f.write('\n')
        #     break
        # break


if __name__ == '__main__':
    n_gpus = sys.argv[1]
    run_ideal_benchmark(n_workers=int(n_gpus))