from sklearn.linear_model import LinearRegression
import numpy as np
from time import time
import sys
import os

base_n_points = 250_000_000

def read_data(path, n_samples, n_features, n_gb, n_samples_per_gb):
    total_file_list = os.listdir(path)
    total_file_list = [path + '/' + tfl for tfl in total_file_list]
    file_list = total_file_list[:n_gb]
    print(file_list)
    print(n_samples, n_samples_per_gb)
    
    if n_features:
        X = np.zeros((n_samples, n_features), dtype='float32', order='C')
        print(X.shape)
        for i in range(len(file_list)):
            X[i * n_samples_per_gb: (i + 1) * n_samples_per_gb, :] = np.load(file_list[i])
    else:
        X = np.zeros((n_samples, ), dtype='float32', order='C')
        for i in range(len(file_list)):
            X[i * n_samples_per_gb: (i + 1) * n_samples_per_gb] = np.load(file_list[i])
    print(X.shape)
    print(X.strides)
    return X


def run_skl_benchmark(n_cores, X_filepath, y_filepath, n_gb, n_features):
    n_points = int(base_n_points * n_gb)
    n_samples = int(n_points / n_features)
    n_samples_per_gb = int(n_samples / n_gb)

    X = read_data(X_filepath, n_samples, n_features, n_gb, n_samples_per_gb)
    y = read_data(y_filepath, n_samples, None, n_gb, n_samples_per_gb)
    
    lr = LinearRegression(n_jobs=n_cores)

    start_fit_time = time()
    lr.fit(X, y)
    end_fit_time = time()
    print("Shape: ", X.shape, ", Fit Time: ", end_fit_time - start_fit_time)
    fit_time = end_fit_time - start_fit_time

    start_pred_time = time()
    preds = lr.predict(X)
    # wait(client.compute(preds))
    end_pred_time = time()
    print("Shape: ", X.shape, ", Predict Time: ", end_pred_time - start_pred_time)
    pred_time = end_pred_time - start_pred_time

    mse = np.mean((y - preds) ** 2)
    print(mse)

    del X, y, preds
    to_write = ','.join(map(str, [n_cores, n_samples, n_features, fit_time, pred_time, mse]))
    with open('/gpfs/fs1/dgala/b_outs/skl_benchmark.csv', 'a') as f:
        f.write(to_write)
        f.write('\n')

if __name__ == '__main__':
    n_cores = int(sys.argv[1])
    X_filepath = sys.argv[2]
    y_filepath = sys.argv[3]
    n_gb = int(sys.argv[4])
    n_features = int(sys.argv[5])
    run_skl_benchmark(n_cores, X_filepath, y_filepath, n_gb, n_features)