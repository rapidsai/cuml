#!/usr/bin/env python
import numpy as np
import os

from cuml.test.utils import array_equal
from cuml.utils.import_utils import has_xgboost

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
    
from cuml import ForestInference

from print_setting_kernel.print_setting_kernel import print_setter

print(print_setter(12, 7).copy_to_host())
print(print_setter(6, 3).copy_to_host())
print(print_setter(10, 4).copy_to_host())

exit(0)

class SKLearnRFClassifier():
    def train(self, X_train, y_train,
                            num_rounds):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(max_depth=5)
        self.model.fit(X_train, y_train)

    def fil(self):
        return ForestInference.load_from_sklearn(
            self.model, algo='NAIVE', storage_type='sparse', output_class=True, threshold=0.5)

n_rows = 10000
n_columns = 10
n_categories = 2
random_state = np.random.RandomState(43210)

# num of iterations for which the model is trained
num_rounds = 15

# create the dataset
X, y = make_classification(n_samples=n_rows,
                           n_features=n_columns,
                           n_informative=int(n_columns/5),
                           n_classes=n_categories,
                           random_state=random_state)
train_size = 0.8

# convert the dataset to np.float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# split the dataset into training and validation splits
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, train_size=train_size)

def try_sk():
    sk = SKLearnRFClassifier()
    sk.train(X_train, y_train, num_rounds)
    fil = sk.fil()
    
    def try_with(low, high):
        x = X_validation[low:high,:]
        print(fil.predict(x).copy_to_host())
        print(fil.predict_proba(x).copy_to_host())
    
    try_with(3, 5)
    try_with(5, 9)
    try_with(11, 15)


try_sk()

from numba import cuda
import rmm

@cuda.jit
def set_and_print(an_array, modulo):
    for i in range(an_array.size):
        an_array[i] = i % modulo
        print(an_array[i])

an_array = rmm.device_array((2, 2), dtype=np.float32)
set_and_print[1, 1](an_array.reshape([4]), 5)
print(an_array.copy_to_host())

