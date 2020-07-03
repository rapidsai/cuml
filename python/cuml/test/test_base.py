# Copyright (c) 2019, NVIDIA CORPORATION.
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

import pytest
import cuml
from cuml.test.utils import small_classification_dataset


def test_base_class_usage():
    base = cuml.Base()
    base.handle.sync()
    base_params = base.get_param_names()
    assert base_params == []
    del base


def test_base_class_usage_with_handle():
    handle = cuml.Handle()
    stream = cuml.cuda.Stream()
    handle.setStream(stream)
    base = cuml.Base(handle=handle)
    base.handle.sync()
    del base


def test_base_hasattr():
    base = cuml.Base()
    # With __getattr__ overriding magic, hasattr should still return
    # True only for valid attributes
    assert hasattr(base, "handle")
    assert not hasattr(base, "somefakeattr")


@pytest.mark.parametrize('datatype', ["float32", "float64"])
@pytest.mark.parametrize('use_integer_n_features', [True, False])
def test_base_n_features_in(datatype, use_integer_n_features):
    X_train, _, _, _ = small_classification_dataset(datatype)
    integer_n_features = 8
    clf = cuml.Base()

    if use_integer_n_features:
        clf._set_n_features_in(integer_n_features)
        assert clf.n_features_in_ == integer_n_features
    else:
        clf._set_n_features_in(X_train)
        assert clf.n_features_in_ == X_train.shape[1]
