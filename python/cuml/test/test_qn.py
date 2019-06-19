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


# todo: add util functions to better compare against precomputed solutions
@pytest.mark.parametrize('loss', ['sigmoid', 'softmax'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('l1_strength', [0.00, 0.01])
@pytest.mark.parametrize('l2_strength', [0.00, 0.01])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_qn(loss, dtype, penalty, l1_strength, l2_strength, fit_intercept):

    tol = 1e-6

    X = np.array(precomputed_X, dtype=dtype)

    if loss == 'sigmoid':
        y = np.array(precomputed_y_log, dtype=dtype)
    else:
        y = np.array(precomputed_y_multi, dtype=dtype)

    qn = cuQN(loss=loss, fit_intercept=fit_intercept, l1_strength=l1_strength,
              l2_strength=l2_strength, tol=1e-8)

    qn.fit(X, y)

    print(qn.objective)
    print(qn.coef_.copy_to_host())

    if loss == 'sigmoid':
        if penalty == 'none' and l1_strength == 0.0 and l2_strength == 0.0:
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
        elif penalty == 'l1' and l2_strength == 0.0:
            if fit_intercept:
                if l1_strength == 0.0:
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
                if l1_strength == 0.0:
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

        elif penalty == 'l2' and l1_strength == 0.0:
            if fit_intercept:
                if l2_strength == 0.0:
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
                if l2_strength == 0.0:
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

        if penalty == 'elasticnet':
            if fit_intercept:
                if l1_strength == 0.0 and l2_strength == 0.0:
                    assert (qn.objective - 0.40263831615448) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.1088872],
                                                            [2.4812558],
                                                            [0.7960136]]),
                                                   decimal=3)
                elif l1_strength == 0.0:
                    assert (qn.objective - 0.43780848383903503) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.5337948],
                                                            [1.678699],
                                                            [0.8060587]]),
                                                   decimal=3)
                elif l2_strength == 0.0:
                    assert (qn.objective - 0.44295936822891235) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.6899368],
                                                            [1.9021575],
                                                            [0.8057671]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.467987984418869) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.3727235],
                                                            [1.4639963],
                                                            [0.79312485]]),
                                                   decimal=3)
            else:
                if l1_strength == 0.0 and l2_strength == 0.0:
                    assert (qn.objective - 0.4317452311515808) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-2.120777],
                                                            [3.056865]]),
                                                   decimal=3)
                elif l1_strength == 0.0:
                    assert (qn.objective - 0.4750209450721741) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.3931049],
                                                            [2.0140104]]),
                                                   decimal=3)

                elif l2_strength == 0.0:
                    assert (qn.objective - 0.4769895672798157) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.6214856],
                                                            [2.3650239]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.5067970156669617) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[-1.2102532],
                                                            [1.752459]]),
                                                   decimal=3)

                print()

    # todo add tests for softmax dtype=np.float64
    # elasticnet for this points converged to different solution
    if loss == 'softmax' and dtype == np.float32:
        if penalty == 'none' and l1_strength == 0.0 and l2_strength == 0.0:
            if fit_intercept:
                assert (qn.objective - 0.007433414924889803) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[15.236361,
                                                         -41.595913,
                                                         -33.55021],
                                                        [-36.607555,
                                                         -13.91267,
                                                         -42.66093],
                                                        [-25.04939,
                                                         -26.793947,
                                                         -31.50192]]),
                                               decimal=3)
            else:
                assert (qn.objective - 0.18794211745262146) < tol
                np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                               np.array([[14.2959795,
                                                         -104.63812,
                                                         -96.41866],
                                                        [-105.31236,
                                                         -170.4887,
                                                         -96.486]]),
                                               decimal=3)
        elif penalty == 'l1' and l2_strength == 0.0:
            if fit_intercept:
                if l1_strength == 0.0:
                    assert (qn.objective - 0.007433414924889803) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[15.236361,
                                                             -41.595913,
                                                             -33.55021],
                                                            [-36.607555,
                                                             -13.91267,
                                                             -42.66093],
                                                            [-25.04939,
                                                             -26.793947,
                                                             -31.50192]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.2925984263420105) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.2279763,
                                                              -2.011927,
                                                              -1.8038181],
                                                             [-3.3828118,
                                                              -0.64903206,
                                                              -3.0688426],
                                                             [-1.6962943,
                                                              -0.8585775,
                                                              -1.1564851]]),
                                                   decimal=3)

            else:
                if l1_strength == 0.0:
                    assert (qn.objective - 0.18794211745262146) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[14.2959795,
                                                             -104.63812,
                                                             -96.41866],
                                                            [-105.31236,
                                                             -170.4887,
                                                             -96.486]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.3777262568473816) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.4765631,
                                                              -1.569497,
                                                              -0.6421711],
                                                             [-2.0787644,
                                                              -1.593922,
                                                              -0.73674846]]),
                                                   decimal=3)

        elif penalty == 'l2' and l1_strength == 0.0:
            if fit_intercept:
                if l2_strength == 0.0:
                    assert (qn.objective - 0.1726059913635254) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[26.228544,
                                                             -40.262405,
                                                             -54.424152],
                                                            [-25.567667,
                                                             -28.038565,
                                                             -50.93895],
                                                            [-17.630503,
                                                             -44.088245,
                                                             -41.725666]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.28578639030456543) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.6702422,
                                                             -1.5495867,
                                                             -1.193351],
                                                            [-2.207053,
                                                             -0.6854614,
                                                             -2.0305414],
                                                            [-1.1746005,
                                                             -0.7992407,
                                                             -1.0034739]]),
                                                   decimal=3)

            else:
                if l2_strength == 0.0:
                    assert (qn.objective - 0.18794211745262146) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[14.2959795,
                                                             -104.63812,
                                                             -96.41866],
                                                            [-105.31236,
                                                             -170.4887,
                                                             -96.486]]),
                                                   decimal=3)

                else:
                    assert (qn.objective - 0.3537392020225525) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.3769588,
                                                             -1.0002015,
                                                             -0.5205092],
                                                            [-1.5185534,
                                                             -1.029575,
                                                             -0.47429192]]),
                                                   decimal=3)

        if penalty == 'elasticnet':
            if fit_intercept:
                if l1_strength == 0.0 and l2_strength == 0.0:
                    assert (qn.objective - 0.007433414924889803) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[15.236361,
                                                             -41.595913,
                                                             -33.55021],
                                                            [-36.607555,
                                                             -13.91267,
                                                             -42.66093],
                                                            [-25.04939,
                                                             -26.793947,
                                                             -31.50192]]),
                                                   decimal=3)
                elif l1_strength == 0.0:
                    assert (qn.objective - 0.28578639030456543) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.6702422,
                                                             -1.5495867,
                                                             -1.193351],
                                                            [-2.207053,
                                                             -0.6854614,
                                                             -2.0305414],
                                                            [-1.1746005,
                                                             -0.7992407,
                                                             -1.0034739]]),
                                                   decimal=3)
                elif l2_strength == 0.0:
                    assert (qn.objective - 0.2925984263420105) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.2279763,
                                                             -2.011927,
                                                             -1.8038181],
                                                            [-3.3828118,
                                                             -0.64903206,
                                                             -3.0688426],
                                                            [-1.6962943,
                                                             -0.8585775,
                                                             -1.1564851]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.34934690594673157) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.1901233,
                                                             -1.2236115,
                                                             -1.0416932],
                                                            [-2.3100038,
                                                             -0.46381754,
                                                             -2.1544967],
                                                            [-1.0984052,
                                                             -0.44855425,
                                                             -0.7347126]]),
                                                   decimal=3)
            else:
                if l1_strength == 0.0 and l2_strength == 0.0:
                    assert (qn.objective - 0.18794211745262146) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[14.2959795,
                                                             -104.63812,
                                                             -96.41866],
                                                            [-105.31236,
                                                             -170.4887,
                                                             -96.486]]),
                                                   decimal=3)
                elif l1_strength == 0.0:
                    assert (qn.objective - 0.3537392020225525) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.3769588,
                                                             -1.0002015,
                                                             -0.5205092],
                                                            [-1.5185534,
                                                             -1.029575,
                                                             -0.47429192]]),
                                                   decimal=3)

                elif l2_strength == 0.0:
                    assert (qn.objective - 0.3777262568473816) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.4765631,
                                                             -1.569497,
                                                             -0.6421711],
                                                            [-2.0787644,
                                                             -1.593922,
                                                             -0.73674846]]),
                                                   decimal=3)
                else:
                    assert (qn.objective - 0.40656331181526184) < tol
                    np.testing.assert_almost_equal(qn.coef_.copy_to_host(),
                                                   np.array([[1.2176441,
                                                             -0.8387626,
                                                             -0.3155345],
                                                            [-1.3095317,
                                                             -0.60578823,
                                                             -0.26777366]]),
                                                   decimal=3)

        else:
            pytest.skip("np.float64 softmax tests in progress.")

    if penalty == "none" and (l1_strength > 0 or l2_strength > 0):
        pytest.skip("`none` penalty does not take l1/l2_strength")


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
