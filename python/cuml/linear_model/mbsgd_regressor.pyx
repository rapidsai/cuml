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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.solvers import SGD

class MBSGDRegressor:

    def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, epochs=1000, tol=1e-3,
                 shuffle=True, learning_rate='constant', eta0=0.0, power_t=0.5,
                 batch_size=32, n_iter_no_change=5, handle=None):
        
        if loss in ['squared_loss']:
            self.loss = loss
        else:
            msg = "loss {!r} is not supported"
            raise TypeError(msg.format(loss))

        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.handle = handle

    def fit(self, X, y):
        self.cu_mbsgd_classifier = SGD(**self.get_params())
        self.cu_mbsgd_classifier.fit(X, y)

    def predict(self, X):
        return self.cu_mbsgd_classifier.predict(X)

    def get_params(self, deep=True):
        params = dict()
        variables = ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept',
                     'epochs', 'tol', 'shuffle', 'learning_rate', 'eta0', 'power_t',
                     'batch_size', 'n_iter_no_change', 'handle']
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        if not params:
            return self
        variables = ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept',
                     'epochs', 'tol', 'shuffle', 'learning_rate', 'eta0', 'power_t',
                     'batch_size', 'n_iter_no_change', 'handle']
        for key, value in params.items():
            if key not in variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)

        return self