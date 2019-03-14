import numpy as np
import cudf
from cuml.solvers import SGD as cumlSGD
import pytest


@pytest.mark.parametrize('lrate', ['constant', 'invscaling', 'adaptive'])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('loss', ['hinge', 'log', 'squared_loss'])
def test_svd(datatype, lrate, input_type, penalty, loss):
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=datatype)
    y = np.array([1, 1, 2, 2], dtype=datatype)
    pred_data = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)
    if input_type == 'dataframe':
        X = cudf.DataFrame()
        X['col1'] = np.asarray([-1, -2, 1, 2], dtype=datatype)
        X['col2'] = np.asarray([-1, -1, 2, 2], dtype=datatype)
        y = cudf.Series(np.array(y, dtype=np.float32))
        pred_data = cudf.DataFrame()
        pred_data['col1'] = np.asarray([3, 2], dtype=datatype)
        pred_data['col2'] = np.asarray([5, 5], dtype=datatype)
    cu_sgd = cumlSGD(learning_rate=lrate, eta0=0.005, epochs=2000,
                     fit_intercept=True, batch_size=2,
                     tol=0.0, penalty=penalty, loss=loss)
    cu_sgd.fit(X, y)
    cu_pred = cu_sgd.predict(pred_data).to_array()
    print("cuML predictions : ", cu_pred)
