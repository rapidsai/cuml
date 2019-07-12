
from cuml.manifold import TSNE as cuml_TSNE
import numpy as np
from sklearn.manifold.t_sne import trustworthiness

from sklearn.datasets import load_digits as data1, load_boston as data2, load_iris as data3, \
	load_breast_cancer as data4, load_diabetes as data5
X1 = data1().data
X2 = data2().data
X3 = data3().data
X4 = data4().data
X5 = data5().data



for i in range(3):
	print("-------------------------------------")
	print("iteration = ", i)

	cuml_tsne = cuml_TSNE(n_components = 2, random_state = i, verbose = 0, learning_rate = 200 + i)
	#print(cuml_tsne)

	Y = cuml_tsne.fit_transform(X1)
	nans = np.sum(np.isnan(Y))
	trust = trustworthiness(X1, Y)
	print("Trust = ", trust)
	if trust < 0.97:
		if (trust < 0.95):
			assert(trust > 0.9)
	del Y

	# Again
	#cuml_tsne = cuml_TSNE(n_components = 2, random_state = i + 1, verbose = 0, learning_rate = 200 + i + 1)
	#print(cuml_tsne)

	Y = cuml_tsne.fit_transform(X2)
	nans = np.sum(np.isnan(Y))
	trust = trustworthiness(X2, Y)
	print("Trust = ", trust)
	if trust < 0.97:
		if (trust < 0.95):
			assert(trust > 0.9)
	del Y

	# Again
	cuml_tsne = cuml_TSNE(n_components = 2, random_state = i + 2, verbose = 0, learning_rate = 200 + i + 2)
	#print(cuml_tsne)

	Y = cuml_tsne.fit_transform(X3)
	nans = np.sum(np.isnan(Y))
	trust = trustworthiness(X3, Y)
	print("Trust = ", trust)
	if trust < 0.97:
		if (trust < 0.95):
			assert(trust > 0.9)
	del Y

	# Again
	#cuml_tsne = cuml_TSNE(n_components = 2, random_state = i + 3, verbose = 0, learning_rate = 200 + i + 3)
	#print(cuml_tsne)

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
	#print(cuml_tsne)

	Y = cuml_tsne.fit_transform(X5)
	nans = np.sum(np.isnan(Y))
	trust = trustworthiness(X5, Y)
	assert (nans == 0)
	print("Trust = ", trust)
	if trust < 0.97:
		if (trust < 0.95):
			assert(trust > 0.9)
	del Y