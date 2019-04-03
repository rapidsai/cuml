# Copyright (c) 2018, NVIDIA CORPORATION.
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

from abc import ABC

import hmmlearn
import cuml
from cuml.hmm.utils.hmm_test_utils import MultinomialHMMSampler, GMMHMMSampler
import numpy as np
import pytest
import time


# @pytest.fixture(params=[40, 50])
# def n_components(request):
#     return request.param
#
#
# @pytest.fixture(params=[40])
# def n_seq(request):
#     return request.param
#
#
# @pytest.fixture(params=[5, 20])
# def n_features(request):
#     return request.param

@pytest.fixture(params=[4])
def n_components(request):
    return request.param


@pytest.fixture(params=[20])
def n_seq(request):
    return request.param


@pytest.fixture(params=[4])
def n_features(request):
    return request.param

@pytest.fixture(params=[1])
def n_mix(request):
    return request.param

@pytest.fixture(params=[2])
def n_dim(request):
    return request.param


class TestHMM(ABC):
    def setup_method(self, method):
        pass

    @staticmethod
    def error(x, y):
        return np.abs(np.max(x - y))


class TestMultinomialHMM(TestHMM):

    def setmethod(self, n_components, n_seq, n_features):
        self.n_components = n_components
        self.n_seq = n_seq
        self.n_features = n_features

        self.hmm_sampler = MultinomialHMMSampler(n_seq=self.n_seq,
                                                 n_components=self.n_components,
                                                 n_features=self.n_features)
        self.precision = "double"
        self.random_state = None
        self.seed = 0

    def _reset(self):
        self.ref_model = hmmlearn.hmm.MultinomialHMM(self.n_components,
                                                     random_state=self.random_state,
                                                     init_params = '')
        self.cuml_model = cuml.hmm.MultinomialHMM(self.n_components,
                                                  precision=self.precision,
                                                  random_state=self.random_state,
                                                  init_params='')

        self.X, self.lengths = self.hmm_sampler.sample_sequences()

        random_state = np.random.randint(50, size=(1))[0]
        self.ref_model = self.hmm_sampler._set_model_parameters(self.ref_model, random_state)
        self.cuml_model= self.hmm_sampler._set_model_parameters(self.cuml_model, random_state)

    # @setup_parameters()
    def test_score_samples(self, n_components, n_seq, n_features):
        self.setmethod(n_components, n_seq, n_features)
        self._reset()

        start = time.time()
        cuml_ll, cuml_posteriors = self.cuml_model.score_samples(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for cuml : ", end - start, "\n")

        start = time.time()
        ref_ll, ref_posteriors = self.ref_model.score_samples(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for hmmlearn : ", end - start, "\n")

        posteriors_err = np.abs(np.max(ref_posteriors - cuml_posteriors))

        assert abs(ref_ll - cuml_ll) < 1e-10
        assert posteriors_err < 1e-10

    def test_decode(self, n_components, n_seq, n_features):
        self.setmethod(n_components, n_seq, n_features)
        self._reset()

        start = time.time()
        cuml_llhd, cuml_state_seq = self.cuml_model.score_samples(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for cuml : ", end - start, "\n")

        start = time.time()
        ref_llhd, ref_state_seq = self.ref_model.score_samples(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for hmmlearn : ", end - start, "\n")

        state_seq_err = np.abs(np.max(ref_state_seq - cuml_state_seq))

        assert abs(cuml_llhd- ref_llhd) < 1e-10
        assert state_seq_err < 1e-10

    # def test_fit(self, n_components, n_seq, n_features):
    #     self.setmethod(n_components, n_seq, n_features)
    #     self._reset()
    #
    #     self.cuml_model.n_iter = 1
    #     self.ref_model.n_iter = 1
    #
    #     self.ref_model.fit(self.X, self.lengths)
    #     self.cuml_model.fit(self.X, self.lengths)
    #
    #     emissionprob_err= self.error(self.ref_model.emissionprob_, self.cuml_model.emissionprob_)
    #     startprob_err = self.error(self.ref_model.startprob_, self.cuml_model.startprob_)
    #     transmat_err = self.error(self.ref_model.transmat_, self.cuml_model.transmat_)
    #
    #     print('transmats')
    #     print(self.ref_model.transmat_)
    #     print(self.cuml_model.transmat_)
    #
    #     assert startprob_err < 1e-10
    #     assert emissionprob_err < 1e-10
    #     assert transmat_err < 1e-10


# class TestGMMHMM(TestHMM):
#     def setmethod(self, n_components, n_seq, n_mix, n_dim):
#         self.n_components = n_components
#         self.n_seq = n_seq
#         self.n_mix = n_mix
#         self.n_dim = n_dim
#
#         self.hmm_sampler = GMMHMMSampler(n_seq=self.n_seq,
#                                          n_components=self.n_components,
#                                          n_mix=n_mix,
#                                          n_dim=n_dim)
#         self.precision = "double"
#         self.random_state = None
#         self.seed = 0
#
#     def _reset(self):
#         self.ref_model = hmmlearn.hmm.GMMHMM(n_components=self.n_components,
#                                              n_mix=self.n_mix,
#                                                      random_state=self.random_state,
#                                                      init_params = '')
#         self.cuml_model = cuml.hmm.GMMHMM(n_components=self.n_components,
#                                              n_mix=self.n_mix,
#                                                   precision=self.precision,
#                                                   random_state=self.random_state,
#                                                   init_params='')
#
#         self.X, self.lengths = self.hmm_sampler.sample_sequences()
#
#         random_state = np.random.randint(50, size=(1))[0]
#         self.ref_model = self.hmm_sampler._set_model_parameters(self.ref_model, random_state)
#         self.cuml_model= self.hmm_sampler._set_model_parameters(self.cuml_model, random_state)
#
#     # @setup_parameters()
#     def test_score_samples(self, n_components, n_seq, n_mix, n_dim):
#         self.setmethod(n_components, n_seq, n_mix, n_dim)
#         self._reset()
#
#         start = time.time()
#         cuml_ll, cuml_posteriors = self.cuml_model.score_samples(self.X, self.lengths)
#         end = time.time()
#         print("\n Elapsed time for cuml : ", end - start, "\n")
#
#         start = time.time()
#         ref_ll, ref_posteriors = self.ref_model.score_samples(self.X, self.lengths)
#         end = time.time()
#         print("\n Elapsed time for hmmlearn : ", end - start, "\n")
#
#         posteriors_err = np.abs(np.max(ref_posteriors - cuml_posteriors))
#
#         assert abs(ref_ll - cuml_ll) < 1e-10
#         assert posteriors_err < 1e-10
#
#     def test_decode(self, n_components, n_seq, n_mix, n_dim):
#         self.setmethod(n_components, n_seq, n_mix, n_dim)
#         self._reset()
#
#         start = time.time()
#         cuml_llhd, cuml_state_seq = self.cuml_model.score_samples(self.X, self.lengths)
#         end = time.time()
#         print("\n Elapsed time for cuml : ", end - start, "\n")
#
#         start = time.time()
#         ref_llhd, ref_state_seq = self.ref_model.score_samples(self.X, self.lengths)
#         end = time.time()
#         print("\n Elapsed time for hmmlearn : ", end - start, "\n")
#
#         state_seq_err = np.abs(np.max(ref_state_seq - cuml_state_seq))
#
#         assert abs(cuml_llhd- ref_llhd) < 1e-10
#         assert state_seq_err < 1e-10