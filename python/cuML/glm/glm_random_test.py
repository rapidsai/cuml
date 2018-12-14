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

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b

def ridge(X, y, alpha=.5):
    coef = 0;
    U, S, V = np.linalg.svd(X, full_matrices=True)
    K = np.zeros(2)
    V2 = np.zeros((2,2))
    U2 = np.zeros((3,2))
    
    for i in range(0, 2):
       K[i] = S[i] * S[i]

    print("K")
    print(K)

    for i in range(0, 2):
       K[i] = K[i] + alpha

    print("K second")
    print(K)

    for i in range(0, 2):
       K[i] = 1 / K[i]

    print("K third")
    print(K)

    for i in range(0, 2):
       V2[:,i] = V[:,i] * K[i]

    print("V2")
    print(V2)

    for i in range(0, 2):
       V2[:,i] = V2[:,i] * S[i]
       U2[:,i] = U[:,i]

    print("V2 second")
    print(V2)

    print("U")
    print(U)

    coef = np.dot(V2, np.transpose(U2))

    print("U secon")
    print(coef)

    coef = np.dot(coef, y)
    
    return coef

X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
y = np.array([0.0, 0.1, 1.0]).astype(np.float32)
alpha=.5

reg_imp = ridge(X, y, alpha)
print(reg_imp)

pred_data = np.array([[0.5, 0.2], [2.0, 1.0]]).astype(np.float32)

reg = linear_model.Ridge(alpha=alpha, fit_intercept=False, normalize=False, solver='svd')
reg.fit(X,y)
pred = reg.predict(pred_data)

print("")
print("Data")
print(X)

print("")
print("Labels")
print(y)

print("")
print("Coefficients")
print(reg.coef_)

print("")
print("Intercept")
print(reg.intercept_)


print("")
print("Pred Data")
print(pred_data)

print("")
print("Pred")
print(pred)

s = np.array([1.0, 0.8, 0.3, 0.0, 1e-16])
print("")
print(s)
print("")
idx = s > 1e-15  # same default value as scipy.linalg.pinv
s_nnz = s[idx][:, np.newaxis]

print("")
print(idx)
print("")

print("")
print(s_nnz)
print("")

