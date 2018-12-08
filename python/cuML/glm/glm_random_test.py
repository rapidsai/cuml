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
from sklearn.linear_model import LinearRegression


X = np.array([[2.0, 5.0], [6.0, 9.0], [2.0, 2.0], [2.0, 3.0]]).astype(np.float32)
y = np.dot(X, np.array([2.0, 7.0]).astype(np.float32)) + 3.0

pred_data = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(np.float32)

reg = LinearRegression(fit_intercept=False, normalize=False).fit(X, y)
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



