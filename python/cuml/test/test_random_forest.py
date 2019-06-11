import pytest
from cuml import RandomForest as curfc
from cuml.test.utils import get_handle
from sklearn.ensemble import RandomForestClassifier as skrfc
import numpy as np
from cuml.test.utils import get_pattern

dataset_names = ['varied', 'aniso', 'blobs', 'noisy_circles',
                 'no_structure']


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_rf_predict_numpy(datatype, name, use_handle):
    pat = get_pattern(name, 100)
    X, y = pat[0]
    unique_labels = np.shape(np.unique(y))[0]
    handle, stream = get_handle(use_handle)

    cuml_model = curfc(max_depth=3, max_leaves=2,
                       max_features=1.0, n_bins=2,
                       split_algo=0, min_rows_per_node=2,
                       n_estimators=1, handle=handle,
                       n_unique_labels=unique_labels)
    cuml_model.fit(X, y)
    cu_labels = cuml_model.predict(X) 
    
    sk_model = skrfc(n_estimators=1, max_depth=2,
                     min_samples_split=2, max_features=1.0)
    sk_model.fit(X, y)
    sk_labels = sk_model.predict(X)
    cuml_model.handle.sync()
    if sk_labels == cu_labels:
        assert 1
