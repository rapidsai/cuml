#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from nltk import stem as nltk_stem
from cuml.preprocessing.text import stem as rapids_stem
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")

MAX_DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_DELAY = 1  # seconds


def ensure_treebank_data():
    """
    Ensure the NLTK treebank dataset is available.
    Tries to download it up to MAX_DOWNLOAD_ATTEMPTS times.
    """
    import nltk

    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            nltk.data.find("corpora/treebank")
            return True

        except LookupError:
            nltk.download("treebank", quiet=True)
            time.sleep(DOWNLOAD_DELAY)

    return False


def get_words():
    """
    Returns list of words from nltk treebank
    If the dataset isn’t available even after MAX_DOWNLOAD_ATTEMPTS attempts,
    issues a warning and then skips the test.
    """

    if not ensure_treebank_data():
        msg = (f"Could not download NLTK treebank dataset after "
               "{MAX_DOWNLOAD_ATTEMPTS} attempts. Skipping test.")

        warnings.warn(msg, UserWarning)
        pytest.skip(msg)

    from nltk.corpus import treebank

    word_ls = []
    for item in treebank.fileids():
        for (word, tag) in treebank.tagged_words(item):
            # assuming the words are already lowered
            word = word.lower()
            word_ls.append(word)

    word_ls = list(set(word_ls))
    return word_ls


def test_same_results():
    word_ls = get_words()
    word_ser = cudf.Series(word_ls)

    nltk_stemmer = nltk_stem.PorterStemmer()
    nltk_stemmed = [nltk_stemmer.stem(word) for word in word_ls]

    cuml_stemmer = rapids_stem.PorterStemmer()
    cuml_stemmed = cuml_stemmer.stem(word_ser)

    assert all(
        [a == b for a, b in zip(nltk_stemmed, cuml_stemmed.to_pandas().values)]
    )
