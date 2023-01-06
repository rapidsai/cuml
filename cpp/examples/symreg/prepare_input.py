# Copyright (c) 2021, NVIDIA CORPORATION.
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

from sklearn.model_selection import train_test_split
from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')

rng = np.random.RandomState(seed=2021)

# Training samples
X_train = rng.uniform(-1, 1, 500).reshape(250, 2)
y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

# Testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

print("Training set has n_rows=%d n_cols=%d" % (X_train.shape))
print("Test set has n_rows=%d n_cols=%d" % (X_test.shape))

train_data = "train_data.txt"
test_data = "test_data.txt"
train_labels = "train_labels.txt"
test_labels = "test_labels.txt"

# Save all datasets in col-major format
np.savetxt(train_data, X_train.T, fmt='%.7f')
np.savetxt(test_data, X_test.T, fmt='%.7f')
np.savetxt(train_labels, y_train, fmt='%.7f')
np.savetxt(test_labels, y_test, fmt='%.7f')

print("Wrote %d values to %s" % (X_train.size, train_data))
print("Wrote %d values to %s" % (X_test.size, test_data))
print("Wrote %d values to %s" % (y_train.size, train_labels))
print("Wrote %d values to %s" % (y_test.size, test_labels))
