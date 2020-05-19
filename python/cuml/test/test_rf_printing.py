# Copyright (c) 2020, NVIDIA CORPORATION.
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
import pytest
import sys
from cuml.ensemble import RandomForestClassifier as RFClassifier
from cuml.test.utils import get_handle
from sklearn.datasets import make_classification

@pytest.mark.parametrize('n_estimators', [5, 10, 20])
@pytest.mark.parametrize('detailed_printing', [True, False])
def test_rf_printing(capfd, n_estimators, detailed_printing):

    X, y = make_classification(n_samples=500, n_features=10,
                               n_clusters_per_class=1, n_informative=5,
                               random_state=94929, n_classes=2)
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Create a handle for the cuml model
    handle, stream = get_handle(True, n_streams=1)

    # Initialize cuML Random Forest classification model
    cuml_model = RFClassifier(handle=handle, max_features=1.0, rows_sample=1.0,
                              n_bins=16, split_algo=0, split_criterion=0,
                              min_rows_per_node=2, seed=23707, n_streams=1,
                              n_estimators=n_estimators, max_leaves = -1,
                              max_depth=16)
    # Train model on the data
    cuml_model.fit(X, y)

    if detailed_printing == True:
        cuml_model.print_detailed()
    else:
        cuml_model.print_summary()

    # Read the captured output
    printed_output = capfd.readouterr().out

    # Test 1: Output is non-zero
    assert '' != printed_output

    # Count the number of trees printed
    tree_count = 0
    for line in printed_output.split('\n'):
        if line.strip().startswith('Tree #'):
            tree_count += 1

    # Test 2: Correct number of trees are printed
    assert n_estimators == tree_count
