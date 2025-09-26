#
# Copyright (c) 2025, NVIDIA CORPORATION.
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

from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding


def test_spectral_embedding_make_blobs():
    """Test SpectralEmbedding with make_blobs dataset."""
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    model = SpectralEmbedding(n_components=2, random_state=42)
    X_embedded = model.fit_transform(X)

    assert X_embedded.shape == (100, 2)
