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

import pytest
import cudf

import cuml
from cuml.hmm.sampler import sample_sequences
import hmmlearn
from cuml.hmm.utils import timer, info
from cuml.hmm.test_utils import *

from cuml.hmm.utils import *


class HMMs :
    def __init__(self):
        pass

    def _reset(self):
        self.cuml = cuml.GMMHMM()
        self.sk = hmmlearn.GMMHMM()

    @reset
    def test_workflow(self):
        print(self.cuml.means_)

    @reset
    def test_score_samples(self, X, lengths):

        cuml_out = self.cuml.score_samples(X, lengths)
        sk_out = self.sk.score_samples(X, lengths)
        return mae(cuml_out, sk_out)


if __name__ == '__main__':

    # X, lengths = sample_sequences()
    Tester = HMMs()
    Tester.test_workflow()