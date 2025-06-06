# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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


"""
The following imports are needed so that we can import those classes
from cuml.feature_extraction.text just like scikit-learn. Do not remove.
"""
from cuml.feature_extraction._tfidf import (  # noqa # pylint: disable=unused-import
    TfidfTransformer,
)
from cuml.feature_extraction._tfidf_vectorizer import (  # noqa # pylint: disable=unused-import
    TfidfVectorizer,
)
from cuml.feature_extraction._vectorizers import (  # noqa # pylint: disable=unused-import
    CountVectorizer,
    HashingVectorizer,
)

__all__ = [
    "CountVectorizer",
    "HashingVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
]
