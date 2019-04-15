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

import cuml


def test_base_class_usage():
    base = cuml.Base()
    base.sync()
    base_params = base.get_param_names()
    assert base_params == []
    del base


def test_base_class_usage_with_handle():
    handle = cuml.Handle()
    stream = cuml.cuda.Stream()
    handle.setStream(stream)
    base = cuml.Base(handle=handle)
    base.sync()
    del base
