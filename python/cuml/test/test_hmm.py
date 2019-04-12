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
from cuml.hmm.utils.sampler import MultinomialHMMSampler, GMMHMMSampler
from cuml.hmm.utils.utils import *
import numpy as np
import pytest
import time

np.set_printoptions(threshold=np.inf)
from cuml.test.temp import fit


@pytest.fixture(params=["double"])
def precision(request):
    return request.param


@pytest.fixture(params=[40])
def n_components(request):
    return request.param


@pytest.fixture(params=[int(1e5)])
def n_seq(request):
    return request.param


@pytest.fixture(params=[40])
def n_features(request):
    return request.param

# @pytest.fixture(params=[2])
# def n_components(request):
#     return request.param
#
#
# @pytest.fixture(params=[30])
# def n_seq(request):
#     return request.param
#
#
# @pytest.fixture(params=[20])
# def n_features(request):
#     return request.param

@pytest.fixture(params=[10])
def n_mix(request):
    return request.param

@pytest.fixture(params=[5])
def n_dim(request):
    return request.param


class TestHMM(ABC):
    def setup_method(self, method):
        pass

    @staticmethod
    def error(x, y):
        return np.abs(np.max(x - y))


class TestMultinomialHMM(TestHMM):

    def setmethod(self, n_components, n_seq, n_features, precision):
        self.n_components = n_components
        self.n_seq = n_seq
        self.n_features = n_features

        seq_len_ref = 15

        self.hmm_sampler = MultinomialHMMSampler(n_seq=self.n_seq,
                                                 n_components=self.n_components,
                                                 n_features=self.n_features,
                                                 seq_len_ref=seq_len_ref)
        self.precision = precision
        self.random_state = None
        self.seed = 0
        if self.precision is "single" :
            self.assert_eps = 1e-2
        if self.precision is "double" :
            self.assert_eps = 1e-7

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

    # def test_score(self, n_components, n_seq, n_features, precision):
    #     self.setmethod(n_components, n_seq, n_features, precision)
    #     self._reset()
    #
    #     start = time.time()
    #     cuml_ll = self.cuml_model.score(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for cuml : ", end - start, "\n")
    #
    #     start = time.time()
    #     ref_ll = self.ref_model.score(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for hmmlearn : ", end - start, "\n")
    #
    #     assert abs(ref_ll - cuml_ll) < self.assert_eps
    #
    # def test_score_samples(self, n_components, n_seq, n_features, precision):
    #     self.setmethod(n_components, n_seq, n_features, precision)
    #     self._reset()
    #
    #     start = time.time()
    #     cuml_ll, cuml_posteriors = self.cuml_model.score_samples(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for cuml : ", end - start, "\n")
    #
    #     start = time.time()
    #     ref_ll, ref_posteriors = self.ref_model.score_samples(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for hmmlearn : ", end - start, "\n")
    #
    #     posteriors_err = np.abs(np.max(ref_posteriors - cuml_posteriors))
    #
    #     assert abs(ref_ll - cuml_ll) < self.assert_eps
    #     assert posteriors_err < self.assert_eps

    # def test_decode(self, n_components, n_seq, n_features, precision):
    #     self.setmethod(n_components, n_seq, n_features, precision)
    #     self._reset()
    #
    #     start = time.time()
    #     cuml_llhd, cuml_state_seq = self.cuml_model.decode(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for cuml : ", end - start, "\n")
    #
    #     start = time.time()
    #     ref_llhd, ref_state_seq = self.ref_model.decode(self.X, self.lengths)
    #     end = time.time()
    #     print("\n Elapsed time for hmmlearn : ", end - start, "\n")
    #
    #     state_seq_err = np.sum(np.abs(cuml_state_seq - ref_state_seq))
    #
    #     assert state_seq_err == 0
    #     assert abs(cuml_llhd- ref_llhd) < self.assert_eps

    def test_fit(self, n_components, n_seq, n_features, precision):
        self.setmethod(n_components, n_seq, n_features, precision)
        self._reset()

        self.cuml_model.n_iter = 1
        self.ref_model.n_iter = 1

        start = time.time()
        self.cuml_model.fit(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for cuml : ", end - start, "\n")

        start = time.time()
        self.ref_model.fit(self.X, self.lengths)
        # stats = fit(self.ref_model, self.X, self.lengths)
        # print(stats["trans"])
        end = time.time()
        print("\n Elapsed time for hmmlearn : ", end - start, "\n")

        emissionprob_err= self.error(self.ref_model.emissionprob_, self.cuml_model.emissionprob_)
        startprob_err = self.error(self.ref_model.startprob_, self.cuml_model.startprob_)
        transmat_err = self.error(self.ref_model.transmat_, self.cuml_model.transmat_)
        #
        # # print("emissions")
        # # print(self.ref_model.emissionprob_)
        # # print(self.cuml_model.emissionprob_)
        # #
        # # print("transmat")
        # # print(self.ref_model.transmat_)
        # # print(self.cuml_model.transmat_)
        #
        assert emissionprob_err < self.assert_eps
        assert startprob_err < self.assert_eps
        assert transmat_err < self.assert_eps

    def test_predict_proba(self, n_components, n_seq, n_features, precision):
        self.setmethod(n_components, n_seq, n_features, precision)
        self._reset()

        start = time.time()
        cuml_probas = self.cuml_model.predict_proba(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for cuml : ", end - start, "\n")

        start = time.time()
        ref_probas = self.ref_model.predict_proba(self.X, self.lengths)
        end = time.time()
        print("\n Elapsed time for hmmlearn : ", end - start, "\n")

        probas_err = np.abs(np.max(ref_probas- cuml_probas))

        # print("nSeq", n_seq)
        # print("nObs", self.cuml_model.nObs)
        # B = self.cuml_model._get_B()
        # idx = np.where(np.abs(ref_probas - cuml_probas )> 1e-7)
        # print(np.unique(idx[0]))
        # # print(cuml_probas)
        # print("\n\n ------------- \n\n")
        # # print(ref_probas)

        assert probas_err < self.assert_eps

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