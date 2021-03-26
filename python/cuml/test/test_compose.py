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

import pytest

from cuml.compose import \
    ColumnTransformer as cuColumnTransformer, \
    make_column_transformer as cu_make_column_transformer, \
    make_column_selector as cu_make_column_selector

from sklearn.compose import \
    ColumnTransformer as skColumnTransformer, \
    make_column_transformer as sk_make_column_transformer, \
    make_column_selector as sk_make_column_selector

from cuml.test.test_preproc_utils import clf_dataset

from cuml.experimental.preprocessing import \
    StandardScaler as cuStandardScaler, \
    Normalizer as cuNormalizer

from sklearn.preprocessing import \
    StandardScaler as skStandardScaler, \
    Normalizer as skNormalizer

from cuml.test.test_preproc_utils import assert_allclose


@pytest.mark.parametrize('remainder', ['drop', 'passthrough'])
@pytest.mark.parametrize('transformer_weights', [None, {'scaler': 2.4,
                                                        'normalizer': 1.8}])
def test_column_transformer(clf_dataset, remainder,  # noqa: F811
                            transformer_weights):
    X_np, X = clf_dataset

    cu_transformers = [
        ("scaler", cuStandardScaler(), [0, 2]),
        ("normalizer", cuNormalizer(), [1, 3])
    ]

    transformer = cuColumnTransformer(transformers=cu_transformers,
                                      remainder=remainder,
                                      transformer_weights=transformer_weights)
    t_X = transformer.fit_transform(X)
    assert type(t_X) == type(X)


    sk_transformers = [
        ("scaler", skStandardScaler(), [0, 2]),
        ("normalizer", skNormalizer(), [1, 3])
    ]

    transformer = skColumnTransformer(transformers=sk_transformers,
                                      remainder=remainder,
                                      transformer_weights=transformer_weights)
    sk_t_X = transformer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)
