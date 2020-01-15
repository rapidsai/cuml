#
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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


from numba import cuda


class RegressorMixin:
    """Mixin class for regression estimators in"""

    _estimator_type = "regressor"

    def score(self, X, y, **kwargs):
        """Scoring function for linear classifiers

        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : [cudf.DataFrame]
            Test samples on which we predict
        y : [cudf.Series, device array, or numpy array]
            Ground truth values for predict(X)

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from cuml.metrics.regression import r2_score
        from cuml.utils import input_to_dev_array

        X_m = input_to_dev_array(X)[0]
        y_m = input_to_dev_array(y)[0]

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None
        return r2_score(y_m,
                        cuda.to_device(self.predict(X_m)),
                        handle=handle)


class ClassifierMixin:
    """Mixin class for classifier estimators in"""

    _estimator_type = "classifier"

    def score(self, X, y, **kwargs):
        """Scoring function for based on mean accuracy.

        Parameters
        ----------
        X : [cudf.DataFrame]
            Test samples on which we predict
        y : [cudf.Series, device array, or numpy array]
            Ground truth values for predict(X)

        Returns
        -------
        score : float
            Accuracy of self.predict(X) wrt. y (fraction where y == pred_y)
        """
        from cuml.metrics.accuracy import accuracy_score
        from cuml.utils import input_to_dev_array

        X_m = input_to_dev_array(X)[0]
        y_m = input_to_dev_array(y)[0]

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None

        return accuracy_score(y_m,
                              cuda.to_device(self.predict(X_m)),
                              handle=handle)
