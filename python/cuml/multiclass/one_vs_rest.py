# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cuml.internals
from cuml.common import input_to_host_array
import sklearn.multiclass

class OneVsRestClassifier(sklearn.multiclass.OneVsRestClassifier):
    """ Wrapper around Sckit-learn's class with the same name. This wrapper
    accepts any array type supported by cuML and converts them to numpy if
    needed to call the corresponding sklearn routine.

    See issue https://github.com/rapidsai/cuml/issues/2876 for more info about
    using Sklearn meta estimators.
    """
    def __init__(self, estimator, *args, n_jobs=None):
        super(OneVsRestClassifier, self).__init__(estimator, *args,
                                                  n_jobs=n_jobs)

    def fit(self, X, y):
        X, _, _, _, _ = input_to_host_array(X)
        y, _, _, _, _ = input_to_host_array(y)
        with using_output_type('numpy'):
            return super(OneVsRestClassifier, self).fit(X, y)

    def predict(self, X):
        # out_type = self.estimator._get_output_type(X)
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.using_output_type('numpy'):
            preds = super(OneVsRestClassifier, self).predict(X)
        return preds

    def decision_function(self, X):
        # out_type = self.estimator._get_output_type(X)
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.using_output_type('numpy'):
            df = super(OneVsRestClassifier, self).decision_function(X)
        return df
