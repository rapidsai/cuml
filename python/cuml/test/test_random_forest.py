import pytest
from cuml import Randomforest as curfc
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
    X = StandardScaler().fit_transform(X)
    print("Calling fit_predict")
    handle, stream = get_handle(use_handle)
    cuml_model = curfc(handle=handle, n_estimators=3, max_depth=2)
    cu_labels = cuml_model.fit_predict(X)
    sk_model = skrfc(eps=3, min_samples=2)
    sk_labels = sk_model.fit_predict(X)
    print(X.shape[0])
    cuml_model.handle.sync()
    pdb.set_trace()
    if sk_labels == cu_labels:
        assert 1
