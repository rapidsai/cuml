
from cuml.manifold import TSNE as cuml_TSNE
import numpy as np

cuml_tsne = cuml_TSNE(n_components = 2, random_state = 4, verbose = 1)
print(cuml_tsne)

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = x_train.reshape((60000,28**2)).astype(np.float32)
y = y_train

Y = cuml_tsne.fit_transform(X)
print(Y[:100,0], Y[:100,1])
del Y

# ERROR:
cuml_tsne = cuml_TSNE(n_components = 2, random_state = 3, verbose = 1)
print(cuml_tsne)

Y = cuml_tsne.fit_transform(X)
print(Y[:100,0], Y[:100,1])
del Y