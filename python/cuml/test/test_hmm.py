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


import hmmlearn
import cuml
from cuml.hmm.utils.hmm_test_utils import MultinomialHMMSampler
import numpy as np
import pytest
import time


@pytest.fixture(params=[10])
def n_components(request):
    return request.param


@pytest.fixture(params=[40])
def n_seq(request):
    return request.param


@pytest.fixture(params=[35])
def n_features(request):
    return request.param


class TestMultinomialHMM:
    def setup_method(self, method):
        pass

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
                                                     )
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

    # def test_fit(self):
    #     self._reset()
    #     self.ref_model.fit(self.X, self.lengths)
    #     self.cuml_model.fit(self.X, self.lengths)


# if __name__ == '__main__':
#     Tester = TestMultinomialHMM()
#     Tester.setup_method(None)
#     Tester.test_score_samples()
