
import pytest
from cuml import Randomforest as curfc 
from cuml.test.utils import get_handle
from sklearn.ensemble import RandomForestClassifier as skrfc
import cudf
import numpy as np
import pdb 
from sklearn.preprocessing import StandardScaler
from cuml.test.utils import fit_predict, get_pattern, clusters_equal

dataset_names = ['noisy_moons', 'varied', 'aniso', 'blobs', 'noisy_circles',
                 'no_structure']


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
def test_rf_predict_numpy(datatype, use_handle):
    pat = get_pattern(name, nrows)
    X, y = pat[0]
    X = StandardScaler().fit_transform(X)
    n_train_rows = int(X.shape[0] *0.8)
    X_train = X[0:n_train_rows,:]
    y_train = y[0:n_train_rows,:]
    X_test = X[n_train_rows:,:]
    print("Calling fit and then predict")
    handle, stream = get_handle(use_handle)
    cuml_model = curfc(handle=handle, n_estimators=3, max_depth=2)
    cuml_model.fit(X_train, y_train)
    cuml_predict = cuml_model.predict(X_test)
    sk_model = skrfc(eps=3, min_samples=2)
    sk_model.fit(X_train, y_train)
    sk_predict = sk_model.predict(X_test)
    print(X.shape[0])
    cuml_model.handle.sync()
    pdb.set_trace()

