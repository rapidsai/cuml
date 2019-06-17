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
import numpy as np

from cuml.solvers import QN as cuQN


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('loss', ['sigmoid', 'softmax'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('l1_ratio', [0.0, 0.01])
@pytest.mark.parametrize('l2_ratio', [0.0, 0.01])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_qn_sigmoid(loss, dtype, penalty, l1_ratio, l2_ratio,
                    fit_intercept):

    tol = 1e-6

    X = np.array(precomputed_X, dtype=dtype)

    if loss == 'sigmoid':
        y = np.array(precomputed_y_log, dtype=dtype)
    else:
        y = np.array(precomputed_y_log, dtype=dtype)

    qn = cuQN(loss=loss, fit_intercept=fit_intercept, l1_ratio=l1_ratio,
              l2_ratio=l2_ratio, tol=1e-8)

    qn.fit(X, y)

    print(qn.objective)
    print(qn.coef_.copy_to_host())

    if loss == 'sigmoid':
        if penalty == 'none' and l1_ratio == 0.0 and l2_ratio == 0.0:
            if fit_intercept:
                assert (qn.objective - 0.40263831615448) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[-2.1088872],
                                                        [2.4812558],
                                                        [0.7960136]]),
                                               decimal=3)
            else:
                assert (qn.objective - 0.4317452311515808) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[-2.120777],
                                                        [3.056865]]),
                                               decimal=3)
        elif penalty == 'l1' and l2_ratio == 0.0:
            if fit_intercept:
                if l1_ratio == 0.0:
                    assert (qn.objective - 0.40263831615448) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.1088872],
                                                            [2.4812558],
                                                            [0.7960136]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.44295936822891235) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.6899368],
                                                            [1.9021575],
                                                            [0.8057671]]),
                                                   decimal=3)

            else:
                if l1_ratio == 0.0:
                    assert (qn.objective - 0.4317452311515808) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.120777],
                                                            [3.056865]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.4769895672798157) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.6214856],
                                                            [2.3650239]]),
                                                   decimal=3)

                # assert False

        elif penalty == 'l2' and l1_ratio == 0.0:
            if fit_intercept:
                if l2_ratio == 0.0:
                    assert (qn.objective - 0.40263831615448) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.1088872],
                                                            [2.4812558],
                                                            [0.7960136]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.43780848383903503) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.5337948],
                                                            [1.678699],
                                                            [0.8060587]]),
                                                   decimal=3)

            else:
                if l2_ratio == 0.0:
                    assert (qn.objective - 0.4317452311515808) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.120777],
                                                            [3.056865]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.4750209450721741) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.3931049],
                                                            [2.0140104]]),
                                                   decimal=3)

    if loss == 'softmax':
        if penalty == 'none' and l1_ratio == 0.0 and l2_ratio == 0.0:
            if fit_intercept:
                assert (qn.objective - 0.0675477385520935) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[-0.2665537],
                                                        [0.32115754],
                                                        [0.65791804]]),
                                               decimal=3)
            else:
                assert (qn.objective - 0.25332343578338623) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[-0.08650934],
                                                        [0.3958893]]),
                                               decimal=3)
        elif penalty == 'l1' and l2_ratio == 0.0:
            if fit_intercept:
                if l1_ratio == 0.0:
                    assert (qn.objective - 0.0675477385520935) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.2665537],
                                                            [0.32115754],
                                                            [0.65791804]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.07325305789709091) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.253401],
                                                            [0.2999552],
                                                            [0.65605354]]),
                                                   decimal=3)

            else:
                if l1_ratio == 0.0:
                    assert (qn.objective - 0.25332343578338623) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.08650934],
                                                            [0.3958893]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.25797709822654724) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.07386708],
                                                            [0.37447292]]),
                                                   decimal=3)

        elif penalty == 'l2' and l1_ratio == 0.0:
            if fit_intercept:
                if l2_ratio == 0.0:
                    assert (qn.objective - 0.40263831615448) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.2665537],
                                                            [0.32115754],
                                                            [0.65791804]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.0684034451842308) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.26288024],
                                                            [0.31470022],
                                                            [0.6574964]]),
                                                   decimal=3)

            else:
                if l2_ratio == 0.0:
                    assert (qn.objective - 0.25332343578338623) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.08650934],
                                                            [0.3958893]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.25412964820861816) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-0.08406641],
                                                            [0.38893977]]),
                                                   decimal=3)

    if penalty == "elasticnet" or (l1_ratio > 0 and l2_ratio > 0):
        pytest.xfail("Elastic Net not supported by QN methods.")

    if penalty == "none" and (l1_ratio > 0 or l2_ratio > 0):
        pytest.skip("`none` penalty does not take l1/l2_ratio")


precomputed_X = [
  [-0.2047076594847130, 0.4789433380575482],
  [-0.5194387150567381, -0.5557303043474900],
  [1.9657805725027142, 1.3934058329729904],
  [0.0929078767437177, 0.2817461528302025],
  [0.7690225676118387, 1.2464347363862822],
  [1.0071893575830049, -1.2962211091122635],
  [0.2749916334321240, 0.2289128789353159],
  [1.3529168351654497, 0.8864293405915888],
  [-2.0016373096603974, -0.3718425371402544],
  [1.6690253095248706, -0.4385697358355719]]


precomputed_y_log = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]


precomputed_y_multi = [2, 2, 0, 3, 3, 0, 0, 0, 1, 0]


precomputed_y_reg = [0.2675836026202781,  -0.0678277759663704,
                     -0.6334027174275105, -0.1018336189077367,
                     0.0933815935886932,  -1.1058853496996381,
                     -0.1658298189619160, -0.2954290675648911,
                     0.7966520536712608, -1.0767450516284769]
