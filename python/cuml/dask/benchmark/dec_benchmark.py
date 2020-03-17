from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, futures_of, performance_report
import dask.array as da
from cuml.dask.decomposition import PCA
from sklearn.datasets import make_spd_matrix
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
import os

base_n_points = 250_000_000

def set_alloc():
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def get_covariance(components_, exp_var, whiten, noise_variance_):
    components_ = cp.asnumpy(components_)
    exp_var = cp.asnumpy(exp_var)
    noise_variance_ = cp.asnumpy(noise_variance_)
    if whiten:
        components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
    exp_var_diff = np.maximum(exp_var - noise_variance_, 0.)
    cov = np.dot(components_.T * exp_var_diff, components_)
    cov.flat[::len(cov) + 1] += noise_variance_  # modify diag inplace
    return cov

def _make_decomposition_data(n_samples, n_features, mean, cov_matrix):
    X = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)
    X = cp.array(X, dtype='float32', order='F')
    return X


def make_decomposition_data(client, workers, n_workers, n_samples, n_features, mean, cov_matrix):
    n_samples_per_worker = int(n_samples / n_workers)
    print(workers, n_workers)

    X = [client.submit(_make_decomposition_data, n_samples_per_worker, n_features, mean, cov_matrix, workers=[workers[i]], key=_make_decomposition_data.__name__ + str(i)) for i in range(n_workers)]

    wait([X])
    print(client.has_what())

    X = [da.from_delayed(dask.delayed(x), meta=cp.zeros(1, dtype=cp.float32),
        shape=(np.nan, n_features),
        dtype=cp.float32) for x in X]

    X = da.concatenate(X, axis=0, allow_unknown_chunksizes=True)
    return X


def run_ideal_benchmark(n_workers, n_gb, n_features, scheduler_file):

    # for n_gb_m in n_gb_data:
    #     for n_features in base_n_features:
    if scheduler_file == 'None':
        cluster = LocalCUDACluster(n_workers=n_workers)
    fit_time = np.zeros(6)
    trans_time = np.zeros(6)
    mean_cov_error = np.zeros(6)
    for i in range(6):
        try:
            n_points = int(base_n_points * n_gb)
            if scheduler_file != 'None':
                client = Client(scheduler_file=scheduler_file)
            else:
                client = Client(cluster)
            client.run(set_alloc)

            workers = list(client.has_what().keys())
            print(workers)

            n_samples = int(n_points / n_features)
            mean = np.zeros(n_features)
            cov_matrix = make_spd_matrix(n_features)

            X = make_decomposition_data(client, workers, n_workers, n_samples, n_features, mean, cov_matrix)
            print(X.compute_chunk_sizes().chunks)
            
            pca = PCA(n_components=n_features, client=client)
            print(type(pca))

            start_fit_time = time()
            pca.fit(X)
            end_fit_time = time()
            print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Fit Time: ", end_fit_time - start_fit_time)
            fit_time[i] = end_fit_time - start_fit_time

            start_trans_time = time()
            trans = pca.transform(X)
            parts = client.sync(extract_arr_partitions, trans, client)
            wait([p for w, p in parts])
            # wait(client.compute(preds))
            end_trans_time = time()
            print("nGPUS: ", n_workers, ", Shape: ", X.shape, ", Transform Time: ", end_trans_time - start_trans_time)
            trans_time[i] = end_trans_time - start_trans_time

            pca_cov = get_covariance(pca.components_, pca.explained_variance_, pca.whiten, pca.noise_variance_)
            mean_cov_error[i] = np.mean(cov_matrix.ravel() - pca_cov.ravel())

            del X, trans

        except Exception as e:
            print(e)
            continue

        finally:
            if 'X' in vars():
                del X
            if 'y' in vars():
                del y
            if 'preds' in vars():
                del preds

            client.close()

    if scheduler_file == 'None':
        cluster.close()
    print("starting write")
    fit_stats = [fit_time[0], np.mean(fit_time[1:]), np.min(fit_time[1:]), np.var(fit_time[1:])]
    trans_stats = [np.mean(trans_time[1:]), np.min(trans_time[1:]), np.var(trans_time[1:]), np.mean(mean_cov_error[1:])]
    to_write = ','.join(map(str, [n_workers, n_samples, n_features] + fit_stats + trans_stats))
    print(to_write)
    with open('/gpfs/fs1/dgala/b_outs/benchmark.csv', 'a') as f:
        f.write(to_write)
        f.write('\n')
    print("ending write")

if __name__ == '__main__':
    n_gpus = int(sys.argv[1])
    n_gb = int(sys.argv[2])
    n_features = int(sys.argv[3])
    scheduler_file = sys.argv[4]
    run_ideal_benchmark(n_gpus, n_gb, n_features, scheduler_file)