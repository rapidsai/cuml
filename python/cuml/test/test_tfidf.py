# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import numpy as np
import cupy as cp
from cuml.feature_extraction.tfidf import TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer as SkTfidfTransfo


@pytest.mark.parametrize('norm', ['l1', 'l2'])
@pytest.mark.parametrize('use_idf', [True, False])
@pytest.mark.parametrize('smooth_idf', [True, False])
@pytest.mark.parametrize('sublinear_tf', [True, False])
def test_tfidf_transformer(norm, use_idf, smooth_idf, sublinear_tf):
    data = np.array([
        [0, 1, 1, 1, 0, 0, 1, 0, 1],
        [0, 2, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 1]
    ])
    data_gpu = cp.array(data)

    tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    sk_tfidf = SkTfidfTransfo(norm=norm, use_idf=use_idf,
                              smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    res = tfidf.fit_transform(data_gpu)
    ref = sk_tfidf.fit_transform(data).todense()

    cp.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize('norm', ['l1', 'l2'])
@pytest.mark.parametrize('use_idf', [True, False])
@pytest.mark.parametrize('smooth_idf', [True, False])
@pytest.mark.parametrize('sublinear_tf', [True, False])
def test_tfidf_transformer_copy(norm, use_idf, smooth_idf, sublinear_tf):
    data_gpu = cp.array([
        [0, 1, 1, 1],
        [0, 2, 0, 1]
    ], dtype=cp.float64, order='F')
    print(data_gpu.__cuda_array_interface__['data'][0])

    tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    res = tfidf.fit_transform(data_gpu, copy=False)

    cp.testing.assert_array_equal(data_gpu, res)
