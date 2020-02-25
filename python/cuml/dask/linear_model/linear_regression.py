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

from cuml.dask.common.input_utils import MGData
from cuml.dask.common.input_utils import to_output
from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import default_client
from dask.distributed import wait
from uuid import uuid1


class LinearRegression(object):
    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    cuML's dask Linear Regression (multi-node multi-gpu) expects dask cuDF
    DataFrame and provides an algorithms, Eig, to fit a linear model.
    And provides an eigendecomposition-based algorithm to fit a linear model.
    (SVD, which is more stable than eig, will be added in an upcoming version.)
    Eig algorithm is usually preferred when the X is a tall and skinny matrix.
    As the number of features in X increases, the accuracy of Eig algorithm
    drops.

    This is an experimental implementation of dask Linear Regresion. It
    supports input X that has more than one column. Single column input
    X will be supported after SVD algorithm is added in an upcoming version.

    Parameters
    -----------
    algorithm : 'eig'
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
    fit_intercept : boolean (default = True)
        LinearRegression adds an additional term c to correct for the global
        mean of y, modeling the reponse as "x * beta + c".
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by its
        L2 norm.
        If False, no scaling will be done.

    Attributes
    -----------
    coef_ : cuDF series, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If fit_intercept_ is False, will be 0.
    """

    def __init__(self, client=None, **kwargs):
        self.client = default_client() if client is None else client
        self.kwargs = kwargs
        self.coef_ = None
        self.intercept_ = None
        self._model_fit = False
        self._consec_call = 0

    @staticmethod
    def _func_create_model(sessionId, datatype, **kwargs):
        try:
            from cuml.linear_model.linear_regression_mg \
               import LinearRegressionMG as cumlLinearRegression
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]
        return cumlLinearRegression(handle=handle, output_type=datatype,
                                    **kwargs)

    @staticmethod
    def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
        return f.fit(data, n_rows, n_cols, partsToSizes, rank)

    @staticmethod
    def _func_predict(f, df):
        res = [f.predict(d) for d in df]
        return res

    #todo: move to utils
    @staticmethod
    def _func_get_idx(f, idx):
        return f[idx]

    def fit(self, X, y, force_colocality=False):
        """
        Fit the model with X and y. If force_colocality is set to True,
        the partitions of X and y will be re-distributed to force the
        co-locality.

        In some cases, data samples and their labels can be distributed
        into different workers by dask. In that case, force_colocality
        param can be set to True to re-arrange the data.

        Usually, you will not need to force co-locality if you pass the
        X and y as follows;

        fit(X["all_the_columns_but_labels"], X["labels"])

        You might want to force co-locality if you pass the X and y as
        follows;

        fit(X, y)

        because dask might have distributed the partitions of X and y
        into different workers.

        Parameters
        ----------
        X : dask cuDF dataframe (n_rows, n_features)
            Features for regression
        y : dask cuDF (n_rows, 1)
            Labels (outcome values)
        force_colocality: boolean (True: re-distributes the partitions
                          of X and y to force the co-locality of
                          the partitions)
        """

        # todo: add check for colocality in case force_colocality=False

        X = self.client.persist(X)
        y = self.client.persist(y)

        data = MGData.colocated(data=(X, y), client=self.client)
        self.datatype = data.datatype

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)
        self.ranks = data.ranks

        n_cols = X.shape[1]

        key = uuid1()
        linear_models = [(w, self.client.submit(
            LinearRegression._func_create_model,
            comms.sessionId,
            self.datatype,
            **self.kwargs,
            workers=[w],
            key="%s-%s" % (key, idx)))
            for idx, w in enumerate(data.workers)]

        key = uuid1()
        linear_fit = dict([(data.worker_info[wf[0]]["r"], self.client.submit(
            LinearRegression._func_fit,
            wf[1],
            data.gpu_futures[wf[0]],
            data.total_rows,
            n_cols,
            data.parts_to_sizes,
            data.worker_info[wf[0]]["r"],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(linear_models)])

        wait(list(linear_fit.values()))
        raise_exception_from_futures(list(linear_fit.values()))

        comms.destroy()

        self.local_model = linear_models[0][1].result()
        self.coef_ = self.local_model.coef_
        self.intercept_ = self.local_model.intercept_

    def predict(self, X):
        """
        Make predictions for X and returns a y_pred.

        Parameters
        ----------
        X : dask cuDF dataframe (n_rows, n_features)

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """
        X = X.persist()

        data = MGData.single(X, client=self.client)

        key = uuid1()
        linear_pred = dict([(wf[0], self.client.submit(
            LinearRegression._func_predict,
            self.local_model,
            data.worker_to_parts[wf[0]],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(data.gpu_futures)])

        raise_exception_from_futures(linear_pred.values())


        # loop to order the futures correctly to build the
        # dask-dataframe/array
        # todo: move this to util file
        results = []
        counters = dict.fromkeys(data.workers, 0)
        for idx, part in enumerate(data.gpu_futures):
            results.append(self.client.submit(
                LinearRegression._func_get_idx,
                linear_pred[part[0]],
                counters[part[0]])
            )
            counters[part[0]] = counters[part[0]] + 1

        return to_output(results, self.datatype)

    def get_param_names(self):
        return list(self.kwargs.keys())
