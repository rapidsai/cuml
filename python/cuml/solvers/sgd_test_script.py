import numpy as np
import cudf
from cuml.solvers import SGD as cumlSGD

X = cudf.DataFrame()
X['col1']=np.array([1,1,2,2],dtype=np.float32)
X['col2']=np.array([1,2,2,3],dtype=np.float32)

print("\n\n***** Running fit *****\n")
print("Input Dataframe:")
print(X)

y = cudf.Series(np.array([6.0, 8.0, 9.0, 11.0], dtype=np.float32))
print("Input Labels:")
print(y)

reg_sgd = cumlSGD(learning_rate='constant', eta0=0.005, epochs=2000, fit_intercept=True, batch_size=2, tol=0.0)
reg_sgd.fit(X, y)

reg = reg_sgd.fit(X,y)
print("Coefficients:")
print(reg.coef_)
print("intercept:")
print(reg.intercept_)

