from sklearn.datasets import make_regression
import numpy as np
import os

base_n_points = 250_000_000
one_gb_multipliers = np.asarray([4], dtype='int')
base_n_features = np.asarray([10], dtype='int')

for one_gb_m in one_gb_multipliers:
    for n_features in base_n_features:
        n_points = base_n_points * one_gb_m
        n_samples = int(n_points / n_features)
        print(n_samples, n_features, int(n_features / 10))
        print(type(n_samples), type(n_features), type(int(n_features / 10)))
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features/10))
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        dir_name = 'data-{}'.format(n_features)
        os.mkdir(dir_name)
        os.mkdir(dir_name + '/X')
        os.mkdir(dir_name + '/y')
        X_subarrs = np.split(X, one_gb_m)
        y_subarrs = np.split(y, one_gb_m)
        for i in range(one_gb_m):
            np.save(dir_name + '/X/{}.npy'.format(i), X_subarrs[i])
            np.save(dir_name + '/y/{}.npy'.format(i), y_subarrs[i])
