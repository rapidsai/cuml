# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

import pytest
from cuml.manifold import UMAP
from cuml.internals import GraphBasedDimRedCallback
from sklearn.datasets import load_digits

digits = load_digits()
data, target = digits.data, digits.target


class CustomCallback(GraphBasedDimRedCallback):
    preprocess_event, epoch_event, train_event = False, 0, False

    def check(self):
        assert(self.preprocess_event)
        assert(self.epoch_event > 10)
        assert(self.train_event)

    def on_preprocess_end(self, embeddings):
        self.preprocess_event = True

    def on_epoch_end(self, embeddings):
        self.epoch_event += 1

    def on_train_end(self, embeddings):
        self.train_event = True


reducer = UMAP(n_components=2, callback=CustomCallback())


@pytest.mark.parametrize('n_components', [2, 4, 8])
def test_internals_api(n_components):
    callback = CustomCallback()
    reducer = UMAP(n_components=n_components, callback=callback)
    reducer.fit(data)
    callback.check()
