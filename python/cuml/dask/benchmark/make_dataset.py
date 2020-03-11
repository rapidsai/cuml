from sklearn.datasets import make_regression
import numpy as np
import os
import sys

base_n_points = 250_000_000
one_gb_multipliers = np.asarray([2], dtype='int')
base_n_features = np.asarray([10], dtype='int')

def make_dataset(gb_data, n_features, data_filepath):
    n_points = base_n_points * gb_data
    n_samples = int(n_points / n_features)

    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features/10))
    X = np.array(X, dtype='float32', order='F')
    y = np.array(y, dtype='float32', order='F')

    dir_name = data_filepath + '/data-{}'.format(n_features)
    os.mkdir(dir_name)
    os.mkdir(dir_name + '/X')
    os.mkdir(dir_name + '/y')
    
    X_subarrs = np.split(X, gb_data)
    y_subarrs = np.split(y, gb_data)
    for i in range(gb_data):
        np.save(dir_name + '/X/{}.npy'.format(i), np.array(X_subarrs[i], order='F'))
        np.save(dir_name + '/y/{}.npy'.format(i), np.array(y_subarrs[i], order='F'))

if __name__ == '__main__':
    gb_data = int(sys.argv[1])
    n_features = int(sys.argv[2])
    data_filepath = sys.argv[3]
    make_dataset(gb_data, n_features, data_filepath)
