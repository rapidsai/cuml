# Copyright (c) 2018, NVIDIA CORPORATION.
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

import cuml
import cudf
import pytest
import numpy as np
import pandas as pd
from cuml.linear_model import Lasso as cuLasso
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from cuml.test.utils import array_equal

@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('selection', ['cyclic', 'random'])

def test_lasso(input_type, selection):
	n_samples = 20
	n_feats = 5
	dtype = np.float64
	train_rows = np.int32(n_samples*0.8)
	X, y = make_regression(n_samples=n_samples, n_features=n_feats, 
		n_informative=n_feats, random_state=0)
	X_test = np.array(X[train_rows:, 0:]).astype(dtype)
	y_train = np.array(y[0: train_rows, ]).astype(dtype)
	y_test = np.array(y[train_rows:, ]).astype(dtype)
	X_train = np.array(X[0: train_rows, :]).astype(dtype)

	sklas = Lasso(alpha=np.array([0.01]), fit_intercept=True, 
			normalize=False, max_iter=1000, 
			selection=selection, tol=1e-10)
	sklas.fit(X_train, y_train)
	sk_predict = sklas.predict(X_test)

	cu_lasso = cuLasso(alpha=np.array([0.01]), fit_intercept=True, 
			normalize=False, max_iter=1000, 
			selection=selection, tol=1e-10)

	if input_type == 'dataframe':
		X_train = pd.DataFrame(
				{'fea%d' %i: X_train[0:, i] for i in range(
					X_train.shape[1])})
		y_train = pd.DataFrame(
				{'fea0': y[0:train_rows, ]})
		X_test = pd.DataFrame(
				{'fea%d' %i: X_test[0:, i] for i in range(
					X_test.shape[1])})
		X_cudf = cudf.DataFrame.from_pandas(X_train) 
		y_cudf = y_train.values
		y_cudf = y_cudf[:, 0]
		y_cudf = cudf.Series(y_cudf) 
		X_cudf_test = cudf.DataFrame.from_pandas(X_test)
		cu_lasso.fit(X_cudf, y_cudf)
		cu_predict = cu_lasso.predict(X_cudf_test).to_array()
		#pdb.set_trace()

	else:
		cu_lasso.fit(X, y)
		cu_predict = cu_lasso.predict(X_test).to_array()
		#pdb.set_trace()

	error_sk = mean_squared_error(y_test, sk_predict)
	error_cu = mean_squared_error(y_test, cu_predict)
	assert array_equal(error_sk, error_cu, 1e-2, with_sign=True)
