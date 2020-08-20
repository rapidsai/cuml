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
from cuml.common.base import Base, ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.solvers import SGD


class MBSGDClassifier(Base, ClassifierMixin):
    """
    Linear models (linear SVM, logistic regression, or linear regression)
    fitted by minimizing a regularized empirical loss with mini-batch SGD.
    The MBSGD Classifier implementation is experimental and and it uses a
    different algorithm than sklearn's SGDClassifier. In order to improve
    the results obtained from cuML's MBSGDClassifier:
    * Reduce the batch size
    * Increase the eta0
    * Increase the number of iterations
    Since cuML is analyzing the data in batches using a small eta0 might
    not let the model learn as much as scikit learn does. Furthermore,
    decreasing the batch size might seen an increase in the time required
    to fit the model.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.linear_model import MBSGDClassifier as cumlMBSGDClassifier
        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series(np.array([1, 1, 2, 2], dtype=np.float32))
        pred_data = cudf.DataFrame()
        pred_data['col1'] = np.asarray([3, 2], dtype=np.float32)
        pred_data['col2'] = np.asarray([5, 5], dtype=np.float32)
        cu_mbsgd_classifier = cumlMBSGClassifier(learning_rate='constant',
                                                 eta0=0.05, epochs=2000,
                                                 fit_intercept=True,
                                                 batch_size=1, tol=0.0,
                                                 penalty='l2',
                                                 loss='squared_loss',
                                                 alpha=0.5)
        cu_mbsgd_classifier.fit(X, y)
        cu_pred = cu_mbsgd_classifier.predict(pred_data).to_array()
        print(" cuML intercept : ", cu_mbsgd_classifier.intercept_)
        print(" cuML coef : ", cu_mbsgd_classifier.coef_)
        print("cuML predictions : ", cu_pred)

    Output:

    .. code-block:: python

        cuML intercept :  0.7150013446807861
        cuML coef :  0    0.27320495
                    1     0.1875956
                    dtype: float32
        cuML predictions :  [1. 1.]


    Parameters
    -----------
    loss : {'hinge', 'log', 'squared_loss'} (default = 'squared_loss')
       'hinge' uses linear SVM

       'log' uses logistic regression

       'squared_loss' uses linear regression

    penalty: {'none', 'l1', 'l2', 'elasticnet'} (default = 'none')
       'none' does not perform any regularization

       'l1' performs L1 norm (Lasso) which minimizes the sum of the abs value
       of coefficients

       'l2' performs L2 norm (Ridge) which minimizes the sum of the square of
       the coefficients

       'elasticnet' performs Elastic Net regularization which is a weighted
       average of L1 and L2 norms

    alpha: float (default = 0.0001)
        The constant value which decides the degree of regularization
    l1_ratio: float (default=0.15)
        The l1_ratio is used only when `penalty = elasticnet`. The value for
        l1_ratio should be `0 <= l1_ratio <= 1`. When `l1_ratio = 0` then the
        `penalty = 'l2'` and if `l1_ratio = 1` then `penalty = 'l1'`
    batch_size: int (default = 32)
        It sets the number of samples that will be included in each batch.
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    epochs : int (default = 1000)
        The number of times the model should iterate through the entire dataset
        during training (default = 1000)
    tol : float (default = 1e-3)
       The training process will stop if current_loss > previous_loss - tol
    shuffle : boolean (default = True)
       True, shuffles the training data after each epoch
       False, does not shuffle the training data after each epoch
    eta0 : float (default = 0.001)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    learning_rate : {'optimal', 'constant', 'invscaling', 'adaptive'}
        (default = 'constant')

        `optimal` option will be supported in a future version

        `constant` keeps the learning rate constant

        `adaptive` changes the learning rate if the training loss or the
        validation accuracy does not improve for `n_iter_no_change` epochs.
        The old learning rate is generally divided by 5
    n_iter_no_change : int (default = 5)
        the number of epochs to train without any imporvement in the model
    output_type : {'input', 'cudf', 'cupy', 'numpy'}, optional
        Variable to control output type of the results and attributes of
        the estimators. If None, it'll inherit the output type set at the
        module level, cuml.output_type. If set, the estimator will override
        the global option for its behavior.

    Notes
    ------
    For additional docs, see `scikitlearn's SGDClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_.
    """

    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, epochs=1000, tol=1e-3,
                 shuffle=True, learning_rate='constant', eta0=0.001,
                 power_t=0.5, batch_size=32, n_iter_no_change=5, handle=None,
                 verbose=False, output_type=None):
        super(MBSGDClassifier, self).__init__(handle=handle,
                                              verbose=verbose,
                                              output_type=output_type)
        self.loss = loss
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
        self.solver_model = SGD(**self.get_params())

    @generate_docstring()
    def fit(self, X, y, convert_dtype=True):
        """
        Fit the model with X and y.

        """
        self._set_n_features_in(X)
        self.solver_model._estimator_type = self._estimator_type
        self.solver_model.fit(X, y, convert_dtype=convert_dtype)
        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        """
        preds = \
            self.solver_model.predictClass(X,
                                           convert_dtype=convert_dtype)

        return preds

    def get_params(self, deep=True):
        """
        Scikit-learn style function that returns the estimator parameters.

        Parameters
        -----------
        deep : boolean (default = True)
        """

        params = dict()
        variables = ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept',
                     'epochs', 'tol', 'shuffle', 'learning_rate', 'eta0',
                     'power_t', 'batch_size', 'n_iter_no_change', 'handle']
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Sklearn style set parameter state to dictionary of params.

        Parameters
        -----------
        params : dict of new params
        """

        if not params:
            return self
        variables = ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept',
                     'epochs', 'tol', 'shuffle', 'learning_rate', 'eta0',
                     'power_t', 'batch_size', 'n_iter_no_change', 'handle']
        for key, value in params.items():
            if key not in variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)

        return self
