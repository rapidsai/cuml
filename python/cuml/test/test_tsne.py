

from cuml.manifold import TSNE as cuml_TSNE
import numpy as np
from sklearn.manifold.t_sne import trustworthiness

import pytest
import pandas as pd
import numpy as np
import cudf


def test_tsne():
	"""
	This tests how TSNE handles a lot of input data across time.
	(1) cuDF DataFrames are passed input
	(2) Numpy arrays are passed in
	(3) Params are changed in the TSNE class
	(4) The class gets re-used across time
	(5) Trustworthiness is checked
	(6) Tests NAN in TSNE output for learning rate explosions
	(7) Tests verbosity
	"""
	from sklearn.datasets import load_digits as data1, load_boston as data2, load_iris as data3, \
	load_breast_cancer as data4, load_diabetes as data5

	X1 = data1().data;		X1_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X1))
	X2 = data2().data;		X2_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X2))
	X3 = data3().data;		X3_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X3))
	X4 = data4().data;		X4_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X4))
	X5 = data5().data;		X5_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X5))


	for i in range(3):
		print("-------------------------------------")
		print("iteration = ", i)

		cuml_tsne = cuml_TSNE(n_components = 2, random_state = i, verbose = 0, learning_rate = 200 + i)

		Y = cuml_tsne.fit_transform(X1_cudf).to_pandas().values
		nans = np.sum(np.isnan(Y))
		trust = trustworthiness(X1, Y)
		print("Trust = ", trust)
		if trust < 0.97:
			if (trust < 0.95):
				assert(trust > 0.9)
		del Y

		# Reuse
		Y = cuml_tsne.fit_transform(X2)
		nans = np.sum(np.isnan(Y))
		trust = trustworthiness(X2, Y)
		print("Trust = ", trust)
		if trust < 0.97:
			if (trust < 0.95):
				assert(trust > 0.9)
		del Y

		# Again
		cuml_tsne = cuml_TSNE(n_components = 2, random_state = i + 2, verbose = 1, learning_rate = 200 + i + 2)

		Y = cuml_tsne.fit_transform(X3_cudf).to_pandas().values
		nans = np.sum(np.isnan(Y))
		trust = trustworthiness(X3, Y)
		print("Trust = ", trust)
		if trust < 0.97:
			if (trust < 0.95):
				assert(trust > 0.9)
		del Y

		# Reuse
		Y = cuml_tsne.fit_transform(X4)
		nans = np.sum(np.isnan(Y))
		trust = trustworthiness(X4, Y)
		print("Trust = ", trust)
		if trust < 0.97:
			if (trust < 0.95):
				assert(trust > 0.9)
		del Y

		# Again
		cuml_tsne = cuml_TSNE(n_components = 2, random_state = i + 4, verbose = 0, learning_rate = 200 + i + 4)

		Y = cuml_tsne.fit_transform(X5_cudf).to_pandas().values
		nans = np.sum(np.isnan(Y))
		trust = trustworthiness(X5, Y)
		assert (nans == 0)
		print("Trust = ", trust)
		if trust < 0.97:
			if (trust < 0.95):
				assert(trust > 0.9)
		del Y

