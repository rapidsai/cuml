import numpy as np
import cudf
from cuml.solvers import SGD as cumlSGD
import pytest
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets
import pandas as pd


@pytest.mark.parametrize('lrate', ['constant', 'invscaling', 'adaptive'])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('loss', ['hinge', 'log', 'squared_loss'])
@pytest.mark.parametrize('nrows', [pytest.param(20, marks=pytest.mark.unit),
                                   pytest.param(500000,
                                                marks=pytest.mark.stress),
                                   pytest.param(5000,
                                                marks=pytest.mark.quality)])
@pytest.mark.parametrize('n_feats', [pytest.param(5, marks=pytest.mark.unit),
                                     pytest.param(1000,
                                                  marks=pytest.mark.stress),
                                     pytest.param(500,
                                                  marks=pytest.mark.quality)])
def test_svd(datatype, lrate, input_type, penalty,
             loss, nrows, n_feats):
    n_samples = nrows
    if n_samples > 5000:
        train_rows = np.int32(n_samples*0.8)
        X, y = make_blobs(n_samples=n_samples,
                          n_features=n_feats, random_state=0)
        X_test = np.array(X[train_rows:, 0:]).astype(datatype)
        X_train = np.array(X[0:train_rows, :]).astype(datatype)
        y_train = np.array(y[0:train_rows, ]).astype(datatype)

    elif n_samples > 100:
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        train_rows = np.int32((np.shape(X)[0])*0.8)
        X_test = np.array(X[train_rows:, 0:]).astype(datatype)
        X_train = np.array(X[0:train_rows, :]).astype(datatype)
        y_train = np.array(y[0:train_rows, ]).astype(datatype)

    else:
        X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]],
                           dtype=datatype)
        y_train = np.array([1, 1, 2, 2], dtype=datatype)
        X_test = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    cu_sgd = cumlSGD(learning_rate=lrate, eta0=0.005, epochs=2000,
                     fit_intercept=True, batch_size=2,
                     tol=0.0, penalty=penalty, loss=loss)

    if input_type == 'dataframe':
        y_train_pd = pd.DataFrame({'fea0': y_train[0:, ]})
        X_train_pd = pd.DataFrame(
                     {'fea%d' % i: X_train[0:, i] for i in range(
                             X_train.shape[1])})
        X_test_pd = pd.DataFrame(
                     {'fea%d' % i: X_test[0:, i] for i in range(
                             X_test.shape[1])})
        X_train = cudf.DataFrame.from_pandas(X_train_pd)
        X_test = cudf.DataFrame.from_pandas(X_test_pd)
        y_train = y_train_pd.values
        y_train = y_train[:, 0]
        y_train = cudf.Series(y_train)

    cu_sgd.fit(X_train, y_train)
    cu_pred = cu_sgd.predict(X_test).to_array()
    print("cuML predictions : ", cu_pred)
