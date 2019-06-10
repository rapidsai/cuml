import pytest
from cuml import RandomForest as curfc
from cuml.test.utils import get_handle
from sklearn.ensemble import RandomForestClassifier as skrfc
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler

from cuml.test.utils import get_pattern

dataset_names = ['noisy_moons', 'varied', 'aniso', 'blobs', 'noisy_circles',
                 'no_structure']


@pytest.mark.parametrize('name', dataset_names)
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_rf_predict_numpy(datatype, name, use_handle):
    pat = get_pattern(name, 100)
    X, y = pat[0]
    print("Calling fit_predict")
    handle, stream = get_handle(use_handle)
    cuml_model = curfc(max_depth=2, max_leaves=2, max_features=1.0, n_bins=4, split_algo=0, min_rows_per_node=2, n_estimators=1, handle=handle)
    cuml_model.fit(X,y)
    pdb.set_trace()
    cu_labels = cuml_model.cross_validate(X,y) 
    
    sk_model = skrfc(n_estimators=1, max_depth=2, min_samples_split=2, max_features=1.0)
    sk_model.fit(X,y)
    sk_labels = sk_model.predict(X)
    print(X.shape[0])
    cuml_model.handle.sync()
    pdb.set_trace()
    if sk_labels == cu_labels:
        assert 1

