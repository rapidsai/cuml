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


class TestMultinomialHMM:
    def setup_method(self, method):
        self.n_components = 2
        self.n_features = 3
        self.hmm_sampler = MultinomialHMMSampler(n_seq=1,
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

        self.ref_model = self.hmm_sampler._set_model_parameters(self.ref_model)
        self.cuml_model= self.hmm_sampler._set_model_parameters(self.cuml_model)

    def test_score_samples(self):
        self._reset()
        ref_ll, ref_posteriors = self.ref_model.score_samples(self.X, self.lengths)
        cuml_ll, cuml_posteriors = self.cuml_model.score_samples(self.X, self.lengths)

        assert abs(ref_ll - cuml_ll) < 1e-10

    def test_decode(self):
        self._reset()
        ref_llhd, ref_state_seq = self.ref_model.score_samples(self.X, self.lengths)
        cuml_llhd, cuml_state_seq = self.cuml_model.score_samples(self.X, self.lengths)

    def test_fit(self):
        self._reset()
        self.ref_model.fit(self.X, self.lengths)
        self.cuml_model.fit(self.X, self.lengths)

    def test_predict_proba(self):
        self._reset()
        self.ref_model.predict_proba(self.X, self.lengths)
        self.cuml_model.predict_proba(self.X, self.lengths)

    def score(self):
        self._reset()
        self.ref_model.score(self.X, self.lengths)
        self.cuml_model.score(self.X, self.lengths)


if __name__ == '__main__':
    Tester = TestMultinomialHMM()
    Tester.setup_method(None)
    Tester.test_score_samples()
