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
from scipy import linalg
from sklearn.linear_model import LinearRegression

X = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0]]).astype(np.float32)
y = np.dot(X, np.array([1.0, 2.0]).astype(np.float32)) + 3.0

U, s, VT = linalg.svd(X)
V = np.transpose(VT)
UT = np.transpose(U)
UT_y = np.matmul(UT[0:2,:], y)
UT_y_s = np.array([0.0, 0.0]).astype(np.float32)
UT_y_s[0] = UT_y[0] / s[0]
UT_y_s[1] = UT_y[1] / s[1]

UT_y_s = np.transpose(UT_y_s)
coef_derived = np.matmul(V, UT_y_s)

reg = LinearRegression(fit_intercept=False).fit(X, y)
pred = reg.predict(np.array([[3.0, 5.0], [2.0, 5.0]]))

print("")
print("X")
print(X)

print("")
print("y")
print(y)

print("")
print("U")
print(U)

print("")
print("UT")
print(UT)

print("")
print("V")
print(V)

print("")
print("s")
print(s)

print("")
print("UT_y")
print(UT_y)

print("")
print("UT_y_s")
print(UT_y_s)

print("")
print("coef_derived")
print(coef_derived)

print("")
print("Coefficients")
print(reg.coef_)

print("")
print("Pred")
print(pred)

tran = np.matmul(X, V)
tran[0,0] = tran[0,0] / s[0]
tran[1,0] = tran[1,0] / s[0]
tran[2,0] = tran[2,0] / s[0]
print("")
print("tran")
print(tran)

