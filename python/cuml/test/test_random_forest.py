import pytest
import numpy as np
from cuml.test.utils import get_handle
from cuml.test.utils import array_equal
from sklearn.datasets import make_classification
from cuml.ensemble import RandomForestClassifier as curfc
from sklearn.ensemble import RandomForestClassifier as skrfc


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_rf_predict_numpy(datatype, use_handle):

    X, y = make_classification(n_samples=100, n_features=40,
                               n_clusters_per_class=1, n_informative=30,
                               random_state=123, n_classes=5)
    y = y.astype(np.int32)
    handle, stream = get_handle(use_handle)

    cuml_model = curfc(max_depth=-1, max_leaves=-1, max_features=1.0,
                       n_bins=4, split_algo=0, min_rows_per_node=2,
                       n_estimators=20, handle=handle)
    cuml_model.fit(X, y)
    cu_predict = cuml_model.predict(X)

    sk_model = skrfc(n_estimators=20, max_depth=None,
                     min_samples_split=2, max_features=1.0)
    sk_model.fit(X, y)
    sk_predict = sk_model.predict(X)

    cuml_model.handle.sync()

    assert array_equal(sk_predict, cu_predict, 1e-1, with_sign=True)
