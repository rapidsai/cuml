import numpy as np
import cudf
from cuml import Ridge as cuRidge

lr = cuRidge(alpha=1.0, fit_intercept=True, normalize = False, solver = 'eig')

X = cudf.DataFrame()
X['col1']=np.array([1,1,2,2],dtype=np.float32)
X['col2']=np.array([1,2,2,3],dtype=np.float32)

print("\n\n***** Running fit *****\n")
print("Input Dataframe:")
print(X)

y = cudf.Series(np.array([6.0, 8.0, 9.0, 11.0], dtype=np.float32))
print("Input Labels:")
print(y)

reg = lr.fit(X,y)
print("Coefficients:")
print(reg.coef_)
print("intercept:")
print(reg.intercept_)

print("\n\n***** Running predict *****\n")
X_new = cudf.DataFrame()
X_new['col1']=np.array([3,2],dtype=np.float32)
X_new['col2']=np.array([5,5],dtype=np.float32)

print("Input Dataframe:")
print(X_new)
#preds = lr.predict(X_new)

#print("Preds:")
#print(preds)
