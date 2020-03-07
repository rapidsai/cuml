from sklearn.datasets import make_regression
import numpy as np
import os
import sys

base_n_points = 250_000_000
one_gb_multipliers = np.asarray([2], dtype='int')
base_n_features = np.asarray([10], dtype='int')

def make_dataset(gb_data, n_features):
    n_points = base_n_points * gb_data
    n_samples = int(n_points / n_features)

    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features/10))
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y, dtype='float32')

    dir_name = 'data-{}'.format(n_features)
    os.mkdir(dir_name)
    os.mkdir(dir_name + '/X')
    os.mkdir(dir_name + '/y')
    
    X_subarrs = np.split(X, gb_data)
    y_subarrs = np.split(y, gb_data)
    for i in range(gb_data):
        np.save(dir_name + '/X/{}.npy'.format(i), X_subarrs[i])
        np.save(dir_name + '/y/{}.npy'.format(i), y_subarrs[i])

if __name__ == '__main__':
    gb_data = int(sys.argv[1])
    n_features = int(sys.argv[2])
    make_dataset(gb_data, n_features)
