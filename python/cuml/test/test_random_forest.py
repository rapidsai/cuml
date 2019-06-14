import pytest
import numpy as np
from cuml.test.utils import get_handle
from cuml.test.utils import array_equal
from sklearn.datasets import make_classification
from cuml.ensemble import RandomForestClassifier as curfc
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_rf_predict_numpy(datatype, use_handle):

    X, y = make_classification(n_samples=1000, n_features=100,
                               n_clusters_per_class=1, n_informative=80,
                               random_state=123, n_classes=5)
    y = y.astype(np.int32)

    X_train = np.asarray(X[0:900, :])
    y_train = np.asarray(y[0:900, ])
    X_test = np.asarray(X[900:, :])
    y_test = np.asarray(y[900:, ])
    handle, stream = get_handle(use_handle)
    cuml_model = curfc(max_features=1.0,
                       n_bins=4, split_algo=0, min_rows_per_node=2,
                       n_estimators=40, handle=handle, max_leaves=-1)
    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_acc = accuracy_score(y_test, cu_predict)
    sk_model = skrfc(n_estimators=40, max_depth=None,
                     min_samples_split=2, max_features=1.0)
    sk_model.fit(X_train, y_train)
    sk_predict = sk_model.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_predict)
    cuml_model.handle.sync()
    assert cu_acc >= (sk_acc - 0.07)
