# Copyright (c) 2020, NVIDIA CORPORATION.
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
import warnings

from cudf import Series
from cuml.common.exceptions import NotFittedError
from cuml.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from functools import partial
import cupy as cp
import numbers
import cudf
from cuml.common.type_utils import CUPY_SPARSE_DTYPES
from cudf.utils.dtypes import min_signed_type


def _get_non_alphanumeric_characters(docs):
    characters = docs.str.character_tokenize().unique()
    non_alpha = characters.str.extract(r'([^\w])', expand=False).dropna()
    return non_alpha.tolist() + ['\n']


def _preprocess(doc, lower=False, remove_non_alphanumeric=False,
                delimiter=' '):
    """Chain together an optional series of text preprocessing steps to
    apply to a document.
    Parameters
    ----------
    doc: cudf.Series[str]
        The string to preprocess
    lower: bool
        Whether to use str.lower to lowercase all of the text
    remove_non_alphanumeric: bool
        Whether or not to remove non-alphanumeric characters.
    Returns
    -------
    doc: cudf.Series[str]
        preprocessed string
    """
    if lower:
        doc = doc.str.lower()
    if remove_non_alphanumeric:
        non_alpha = _get_non_alphanumeric_characters(doc)
        delimiter_code = ord(delimiter)
        translation_table = {ord(char): delimiter_code for char in non_alpha}
        doc = doc.str.translate(translation_table)
    return doc


class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    def _remove_stop_words(self, doc):
        """Remove stop words only if needed."""
        if self.analyzer == 'word' and self.stop_words is not None:
            stop_words = Series(self._get_stop_words())
            doc = doc.str.replace_tokens(stop_words,
                                         replacements=self.delimiter,
                                         delimiter=self.delimiter)
        return doc

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization.

        If analyzer == 'word' and stop_words is not None, stop words are
        removed from the input documents after preprocessing.

        Returns
        -------
        preprocessor: callable
              A function to preprocess the text before tokenization.
        """
        if self.preprocessor is not None:
            preprocess = self.preprocessor
        else:
            remove_non_alpha = self.analyzer == 'word'
            preprocess = partial(_preprocess, lower=self.lowercase,
                                 remove_non_alphanumeric=remove_non_alpha,
                                 delimiter=self.delimiter)
        return lambda doc: self._remove_stop_words(preprocess(doc))

    def _get_stop_words(self):
        """Build or fetch the effective stop words list.
        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        if self.stop_words == "english":
            return list(ENGLISH_STOP_WORDS)
        elif isinstance(self.stop_words, str):
            raise ValueError("not a built-in stop list: %s" % self.stop_words)
        elif self.stop_words is None:
            return None
        else:  # assume it's a collection
            return list(self.stop_words)

    def get_char_ngrams(self, ngram_size, str_series, doc_id_sr):
        """
        Handles ngram generation for characters analyzers.

        When analyzer is 'char_wb', we generate ngrams within word boundaries,
        meaning we need to first tokenize and pad each token with a delimiter.
        """
        if self.analyzer == 'char_wb' and ngram_size != 1:
            tokens = str_series.str.tokenize(self.delimiter)

            padding = Series(self.delimiter).repeat(len(tokens))
            tokens = padding.str.cat(tokens.str.cat(padding))
            tokens = tokens.reset_index(drop=True)

            ngram_sr = tokens.str.character_ngrams(n=ngram_size)

            token_count = str_series.str.token_count(self.delimiter)
            doc_id_df = cudf.DataFrame({
                'doc_id': doc_id_sr.repeat(token_count).reset_index(drop=True),
                # formula to count ngrams given number of letters per token:
                'ngram_count': tokens.str.len() - (ngram_size - 1)
            })
            ngram_count = doc_id_df.groupby('doc_id').sum()['ngram_count']
            return ngram_sr, ngram_count, token_count

        if ngram_size == 1:
            token_count = str_series.str.len()
            ngram_sr = str_series.str.character_tokenize()
        elif self.analyzer == 'char':
            token_count = str_series.str.len()
            ngram_sr = str_series.str.character_ngrams(n=ngram_size)

        ngram_count = token_count - (ngram_size - 1)

        return ngram_sr, ngram_count, token_count

    def get_ngrams(self, str_series, ngram_size, doc_id_sr):
        """
        This returns the ngrams for the string series
        Parameters
        ----------
        str_series : (cudf.Series)
            String series to tokenize
        ngram_size : int
            Gram level to get (1 for unigram, 2 for bigram etc)
        doc_id_sr : cudf.Series
            Int series containing documents ids
        """
        if self.analyzer == 'word':
            token_count_sr = str_series.str.token_count(self.delimiter)
            ngram_sr = str_series.str.ngrams_tokenize(n=ngram_size,
                                                      separator=" ",
                                                      delimiter=self.delimiter)
            # formula to count ngrams given number of tokens x per doc: x-(n-1)
            ngram_count = token_count_sr - (ngram_size - 1)
        else:
            ngram_sr, ngram_count, token_count_sr = self.get_char_ngrams(
                ngram_size, str_series, doc_id_sr
            )

        not_empty_docs = token_count_sr > 0
        doc_id_sr = doc_id_sr[not_empty_docs]
        ngram_count = ngram_count[not_empty_docs]

        doc_id_sr = doc_id_sr.repeat(ngram_count).reset_index(drop=True)
        tokenized_df = cudf.DataFrame()
        tokenized_df["doc_id"] = doc_id_sr
        tokenized_df["token"] = ngram_sr
        return tokenized_df

    def _create_tokenized_df(self, docs):
        """Creates a tokenized DataFrame from a string Series.

        Each row describes the token string and the corresponding document id.
        """
        min_n, max_n = self.ngram_range

        doc_id = cp.arange(start=0, stop=len(docs), dtype=cp.int32)
        doc_id = Series(doc_id)

        tokenized_df_ls = [
            self.get_ngrams(docs, n, doc_id)
            for n in range(min_n, max_n + 1)
        ]

        tokenized_df = cudf.concat(tokenized_df_ls)
        tokenized_df = tokenized_df.reset_index(drop=True)

        return tokenized_df

    def _insert_zeros(self, ary, zero_indices):
        """Create a new array of len(ary + zero_indices) where zero_indices
        indicates indexes of 0s in the new array. Ary is used to fill the rest.

        Example:
            _insert_zeros([1, 2, 3], [1, 3]) => [1, 0, 2, 0, 3]
        """
        if len(zero_indices) == 0:
            return ary.values

        new_ary = cp.zeros((len(ary) + len(zero_indices)), dtype=cp.int32)

        # getting mask of non-zeros
        data_mask = ~cp.in1d(cp.arange(0, len(new_ary)), zero_indices)

        new_ary[data_mask] = ary
        return new_ary

    def _compute_empty_doc_ids(self, count_df, n_doc):
        """
        Compute empty docs ids using the remaining docs, given the total number
        of documents.
        """
        remaining_docs = count_df['doc_id'].unique()
        dtype = min_signed_type(n_doc)
        doc_ids = cudf.DataFrame(data={'all_ids': cp.arange(0, n_doc,
                                                            dtype=dtype)},
                                 dtype=dtype)

        empty_docs = doc_ids - doc_ids.iloc[remaining_docs]
        empty_ids = empty_docs[empty_docs['all_ids'].isnull()].index.values
        return empty_ids

    def _create_csr_matrix_from_count_df(self, count_df, empty_doc_ids, n_doc):
        """Create a sparse matrix from the count of tokens by document"""
        n_features = len(self.vocabulary_)

        data = count_df["count"].values
        indices = count_df["token"].values

        doc_token_counts = count_df["doc_id"].value_counts().reset_index()
        doc_token_counts = doc_token_counts.rename(
            {"doc_id": "token_counts", "index": "doc_id"}, axis=1
        ).sort_values(by="doc_id")
        token_counts = self._insert_zeros(doc_token_counts["token_counts"],
                                          empty_doc_ids)
        indptr = token_counts.cumsum()
        indptr = cp.pad(indptr, (1, 0), "constant")

        return cp.sparse.csr_matrix(
            arg1=(data, indices, indptr), dtype=self.dtype,
            shape=(n_doc, n_features)
        )

    def _validate_params(self):
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        msg = ""
        if min_n < 1:
            msg += "lower boundary must be >= 1. "
        if min_n > max_m:
            msg += "lower boundary larger than the upper boundary. "
        if msg != "":
            msg = f"Invalid value for ngram_range={self.ngram_range} {msg}"
            raise ValueError(msg)

    def _warn_for_unused_params(self):
        if self.analyzer != 'word' and self.stop_words is not None:
            warnings.warn("The parameter 'stop_words' will not be used"
                          " since 'analyzer' != 'word'")


def _document_frequency(X):
    """Count the number of non-zero values for each feature in X."""
    doc_freq = (
        X[["token", "doc_id"]]
        .groupby(["token"])
        .count()
    )
    return doc_freq["doc_id"].values


def _term_frequency(X):
    """Count the number of occurrences of each term in X."""
    term_freq = (
        X[["token", "count"]]
        .groupby(["token"])
        .sum()
    )
    return term_freq["count"].values


class CountVectorizer(_VectorizerMixin):
    """Convert a collection of text documents to a matrix of token counts

    If you do not provide an a-priori dictionary then the number of features
    will be equal to the vocabulary size found by analyzing the data.

    Parameters
    ----------
    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the input documents.
        If None, no stop words will be used. max_df can be set to a value
        to automatically detect and filter stop words based on intra corpus
        document frequency of terms.
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
    analyzer : string, {'word', 'char', 'char_wb'}
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : cudf.Series, optional
        If not given, a vocabulary is determined from the input documents.
    binary : boolean, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    delimiter : str, whitespace by default
        String used as a replacement for stop words if stop_words is not None.
        Typically the delimiting character between words is a good choice.

    Attributes
    ----------
    vocabulary_ : cudf.Series[str]
        Array mapping from feature integer indices to feature name.
    stop_words_ : cudf.Series[str]
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    """

    def __init__(self, input=None, encoding=None, decode_error=None,
                 strip_accents=None, lowercase=True, preprocessor=None,
                 tokenizer=None, stop_words=None, token_pattern=None,
                 ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=cp.float32, delimiter=' '):
        self.preprocessor = preprocessor
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if not isinstance(max_features, int) or max_features <= 0:
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.delimiter = delimiter
        if dtype not in CUPY_SPARSE_DTYPES:
            msg = f"Expected dtype in {CUPY_SPARSE_DTYPES}, got {dtype}"
            raise ValueError(msg)

        sklearn_params = {"input": input,
                          "encoding": encoding,
                          "decode_error": decode_error,
                          "strip_accents": strip_accents,
                          "tokenizer": tokenizer,
                          "token_pattern": token_pattern}
        self._check_sklearn_params(analyzer, sklearn_params)

    def _count_vocab(self, tokenized_df):
        """Count occurrences of tokens in each document."""
        # Transform string tokens into token indexes from 0 to len(vocab)
        # The indexes are based on lexicographical ordering.
        tokenized_df['token'] = tokenized_df['token'].astype('category')
        tokenized_df['token'] = tokenized_df['token'].cat.set_categories(
            self.vocabulary_
        )._column.codes

        # Count of each token in each document
        count_df = (
            tokenized_df[["doc_id", "token"]]
            .groupby(["doc_id", "token"])
            .size()
            .reset_index()
            .rename({0: "count"}, axis=1)
        )

        return count_df

    def _filter_and_renumber(self, df, keep_values, column):
        """Filter dataframe to keep only values from column matching
        keep_values."""
        df[column] = (
            df[column].astype('category')
            .cat.set_categories(keep_values)
            ._column.codes
        )
        df = df.dropna(subset=column)
        return df

    def _limit_features(self, count_df, vocab, high, low, limit):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        Sets self.vocabulary_ and self.stop_words_ with the new values.
        """
        if high is None and low is None and limit is None:
            self.stop_words_ = None
            return count_df

        document_frequency = _document_frequency(count_df)

        mask = cp.ones(len(document_frequency), dtype=bool)
        if high is not None:
            mask &= document_frequency <= high
        if low is not None:
            mask &= document_frequency >= low
        if limit is not None and mask.sum() > limit:
            term_frequency = _term_frequency(count_df)
            mask_inds = (-term_frequency[mask]).argsort()[:limit]
            new_mask = cp.zeros(len(document_frequency), dtype=bool)
            new_mask[cp.where(mask)[0][mask_inds]] = True
            mask = new_mask

        keep_idx = cp.where(mask)[0].astype(cp.int32)
        keep_num = keep_idx.shape[0]

        if keep_num == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")

        if len(vocab) - keep_num != 0:
            count_df = self._filter_and_renumber(count_df, keep_idx, 'token')

        self.stop_words_ = vocab[~mask].reset_index(drop=True)
        self.vocabulary_ = vocab[mask].reset_index(drop=True)

        return count_df

    def _preprocess(self, raw_documents):
        preprocess = self.build_preprocessor()
        return preprocess(raw_documents)

    def fit(self, raw_documents):
        """Build a vocabulary of all tokens in the raw documents.

       Parameters
       ----------
       raw_documents : cudf.Series
           A Series of string documents

       Returns
       -------
       self
       """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents):
        """Build the vocabulary and return document-term matrix.

        Equivalent to .fit(X).transform(X) but preprocess X only once.

        Parameters
        ----------
        raw_documents : cudf.Series
           A Series of string documents

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Document-term matrix.
        """
        self._warn_for_unused_params()
        self._validate_params()

        self._fixed_vocabulary = self.vocabulary is not None

        docs = self._preprocess(raw_documents)
        n_doc = len(docs)

        tokenized_df = self._create_tokenized_df(docs)

        if self._fixed_vocabulary:
            self.vocabulary_ = self.vocabulary
        else:
            self.vocabulary_ = tokenized_df["token"].unique()

        count_df = self._count_vocab(tokenized_df)

        if not self._fixed_vocabulary:
            max_doc_count = (self.max_df
                             if isinstance(self.max_df, numbers.Integral)
                             else self.max_df * n_doc)
            min_doc_count = (self.min_df
                             if isinstance(self.min_df, numbers.Integral)
                             else self.min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            count_df = self._limit_features(count_df, self.vocabulary_,
                                            max_doc_count,
                                            min_doc_count,
                                            self.max_features)

        empty_doc_ids = self._compute_empty_doc_ids(count_df, n_doc)

        X = self._create_csr_matrix_from_count_df(count_df, empty_doc_ids,
                                                  n_doc)
        if self.binary:
            X.data.fill(1)
        return X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : cudf.Series
           A Series of string documents

        Returns
        -------
        X : cupy csr array of shape (n_samples, n_features)
            Document-term matrix.
        """
        if not hasattr(self, 'vocabulary_'):
            if self.vocabulary is not None:
                self.vocabulary_ = self.vocabulary
            else:
                raise NotFittedError()

        docs = self._preprocess(raw_documents)
        n_doc = len(docs)
        tokenized_df = self._create_tokenized_df(docs)
        count_df = self._count_vocab(tokenized_df)
        empty_doc_ids = self._compute_empty_doc_ids(count_df, n_doc)
        X = self._create_csr_matrix_from_count_df(count_df, empty_doc_ids,
                                                  n_doc)
        if self.binary:
            X.data.fill(1)
        return X

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        X_inv : list of cudf.Series of shape (n_samples,)
            List of Series of terms.
        """
        vocab = Series(self.vocabulary_)
        return [vocab[X[i, :].indices] for i in range(X.shape[0])]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name.
        Returns
        -------
        feature_names : Series
            A list of feature names.
        """
        return self.vocabulary_

    def _check_sklearn_params(self, analyzer, sklearn_params):
        if callable(analyzer):
            raise ValueError("cuML does not support callable analyzer,"
                             " please refer to the cuML documentation for"
                             " more information.")

        for key, vals in sklearn_params.items():
            if vals is not None:
                raise TypeError("The Scikit-learn variable", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information.")
