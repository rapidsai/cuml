from cuml.metrics import log_loss
import numpy as np
x = log_loss(np.array([1, 0, 0, 1]),np.array([[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
print(x)
