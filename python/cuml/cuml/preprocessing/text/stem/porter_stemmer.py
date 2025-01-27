#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from .porter_stemmer_utils.measure_utils import (
    has_positive_measure,
    measure_gt_n,
    measure_eq_n,
)
from .porter_stemmer_utils.len_flags_utils import (
    len_eq_n,
    len_gt_n,
)
from .porter_stemmer_utils.consonant_vowel_utils import (
    contains_vowel,
    is_consonant,
)
from .porter_stemmer_utils.porter_stemmer_rules import (
    ends_with_suffix,
    ends_with_double_constant,
    last_char_not_in,
    last_char_in,
    ends_cvc,
)
from .porter_stemmer_utils.suffix_utils import (
    get_stem_series,
    get_str_replacement_series,
    replace_suffix,
)
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")


# Implementation based on nltk//stem/porter.html
# https://www.nltk.org/_modules/nltk/stem/porter.html
class PorterStemmer:
    """
    A word stemmer based on the Porter stemming algorithm.

    Porter, M. "An algorithm for suffix stripping."
    Program 14.3 (1980): 130-137.

    See http://www.tartarus.org/~martin/PorterStemmer/ for the homepage
    of the algorithm.

    Martin Porter has endorsed several modifications to the Porter
    algorithm since writing his original paper, and those extensions are
    included in the implementations on his website. Additionally, others
    have proposed further improvements to the algorithm, including NLTK
    contributors. Only below mode is supported currently
    PorterStemmer.NLTK_EXTENSIONS

    - Implementation that includes further improvements devised by
      NLTK contributors or taken from other modified implementations
      found on the web.

    Parameters
    ----------
        mode: Modes of stemming (Only supports (NLTK_EXTENSIONS) currently)
              default("NLTK_EXTENSIONS")

    Examples
    --------

    .. code-block:: python

        >>> import cudf
        >>> from cuml.preprocessing.text.stem import PorterStemmer
        >>> stemmer = PorterStemmer()
        >>> word_str_ser =  cudf.Series(['revival','singing','adjustable'])
        >>> print(stemmer.stem(word_str_ser))
        0     reviv
        1      sing
        2    adjust
        dtype: object

    """

    def __init__(self, mode="NLTK_EXTENSIONS"):
        if mode != "NLTK_EXTENSIONS":
            raise ValueError(
                "Only PorterStemmer.NLTK_EXTENSIONS is supported currently"
            )
        self.mode = mode

    def stem(self, word_str_ser):
        """
        Stem Words using Porter stemmer

        Parameters
        ----------
        word_str_ser : cudf.Series
            A string series of words to stem

        Returns
        -------
        stemmed_ser : cudf.Series
            Stemmed words strings series
        """
        # this is only for NLTK_EXTENSIONS
        # remove the length condition for original algorithm
        # do not stem is len(word) <= 2:
        can_replace_mask = len_gt_n(word_str_ser, 2)

        word_str_ser = word_str_ser.str.lower()

        word_str_ser, can_replace_mask = map_irregular_forms(
            word_str_ser, can_replace_mask
        )

        # apply step 1
        word_str_ser = self._step1a(word_str_ser, can_replace_mask)
        word_str_ser = self._step1b(word_str_ser, can_replace_mask)
        word_str_ser = self._step1c(word_str_ser, can_replace_mask)

        # apply step 2
        word_str_ser = self._step2(word_str_ser, can_replace_mask)

        # apply step 3
        word_str_ser = self._step3(word_str_ser, can_replace_mask)

        # apply step 4
        word_str_ser = self._step4(word_str_ser, can_replace_mask)

        # apply step 5
        word_str_ser = self._step5a(word_str_ser, can_replace_mask)
        word_str_ser = self._step5b(word_str_ser, can_replace_mask)

        return word_str_ser

    def _step1a(self, word_str_ser, can_replace_mask=None):
        """Implements Step 1a from "An algorithm for suffix stripping"

        From the paper:

            SSES -> SS                         caresses  ->  caress
            IES  -> I                          ponies    ->  poni
                                               ties      ->  ti
                                               (### this is for original impl)
            SS   -> SS                         caress    ->  caress
            S    ->                            cats      ->  cat
        """
        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        # this NLTK-only rule extends the original algorithm, so
        # that 'flies'->'fli' but 'dies'->'die' etc
        # ties -> tie
        if self.mode == "NLTK_EXTENSIONS":
            # equivalent to
            # word.endswith('ies') and len(word) == 4:
            suffix_mask = ends_with_suffix(word_str_ser, "ies")
            len_mask = len_eq_n(word_str_ser, 4)

            condition_mask = suffix_mask & len_mask

            valid_mask = can_replace_mask & condition_mask
            word_str_ser = replace_suffix(
                word_str_ser, "ies", "ie", valid_mask
            )

            # update can replace mask
            can_replace_mask &= ~condition_mask

        return apply_rule_list(
            word_str_ser,
            [
                ("sses", "ss", None),  # SSES -> SS
                ("ies", "i", None),  # IES  -> I
                ("ss", "ss", None),  # SS   -> SS
                ("s", "", None),  # S    ->
            ],
            can_replace_mask,
        )[0]

    def _step1b(self, word_str_ser, can_replace_mask=None):
        """Implements Step 1b from "An algorithm for suffix stripping"

        From the paper:

            (m>0) EED -> EE                    feed      ->  feed
                                            agreed    ->  agree
            (*v*) ED  ->                       plastered ->  plaster
                                            bled      ->  bled
            (*v*) ING ->                       motoring  ->  motor
                                            sing      ->  sing

        If the second or third of the rules in Step 1b is successful,
        the following is done:

            AT -> ATE                       conflat(ed)  ->  conflate
            BL -> BLE                       troubl(ed)   ->  trouble
            IZ -> IZE                       siz(ed)      ->  size
            (*d and not (*L or *S or *Z))
            -> single letter
                                            hopp(ing)    ->  hop
                                            tann(ed)     ->  tan
                                            fall(ing)    ->  fall
                                            hiss(ing)    ->  hiss
                                            fizz(ed)     ->  fizz
            (m=1 and *o) -> E               fail(ing)    ->  fail
                                            fil(ing)     ->  file

        The rule to map to a single letter causes the removal of one of
        the double letter pair. The -E is put back on -AT, -BL and -IZ,
        so that the suffixes -ATE, -BLE and -IZE can be recognised
        later. This E may be removed in step 4.
        """

        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        # this NLTK-only block extends the original algorithm, so that
        # 'spied'->'spi' but 'died'->'die' etc
        if self.mode == "NLTK_EXTENSIONS":
            # word.endswith('ied'):
            suffix_mask = ends_with_suffix(word_str_ser, "ied")
            len_mask = len_eq_n(word_str_ser, 4)

            condition_mask = suffix_mask & len_mask

            valid_mask = can_replace_mask & condition_mask
            word_str_ser = replace_suffix(
                word_str_ser, "ied", "ie", valid_mask
            )

            # update can replace mask
            can_replace_mask &= ~condition_mask

            condition_mask = suffix_mask
            valid_mask = can_replace_mask & condition_mask
            word_str_ser = replace_suffix(word_str_ser, "ied", "i", valid_mask)

            # update can replace mask
            can_replace_mask &= ~condition_mask

        # (m>0) EED -> EE
        # if suffix ==eed we stop processing
        # to be consistent with nltk
        suffix_mask = ends_with_suffix(word_str_ser, "eed")
        valid_mask = suffix_mask & can_replace_mask

        stem = replace_suffix(word_str_ser, "eed", "", valid_mask)
        measure_mask = measure_gt_n(stem, 0)

        valid_mask = measure_mask & suffix_mask & can_replace_mask
        # adding ee series to stem
        word_str_ser = replace_suffix(word_str_ser, "eed", "ee", valid_mask)

        # to be consistent with nltk we dont replace
        # if word.endswith('eed') we stop proceesing
        can_replace_mask &= ~suffix_mask

        # rule 2
        #    (*v*) ED  ->   plastered ->  plaster
        #                   bled      ->  bled

        ed_suffix_mask = ends_with_suffix(word_str_ser, "ed")
        intermediate_stem = replace_suffix(
            word_str_ser, "ed", "", ed_suffix_mask & can_replace_mask
        )
        vowel_mask = contains_vowel(intermediate_stem)

        rule_2_mask = vowel_mask & ed_suffix_mask & can_replace_mask

        # rule 3

        #    (*v*) ING ->  motoring  ->  motor
        #                   sing      ->  sing
        ing_suffix_mask = ends_with_suffix(word_str_ser, "ing")
        intermediate_stem = replace_suffix(
            word_str_ser, "ing", "", ing_suffix_mask & can_replace_mask
        )
        vowel_mask = contains_vowel(intermediate_stem)
        rule_3_mask = vowel_mask & ing_suffix_mask & can_replace_mask

        rule_2_or_rule_3_mask = rule_2_mask | rule_3_mask

        # replace masks only if rule_2_or_rule_3_mask
        intermediate_stem_1 = replace_suffix(
            word_str_ser, "ed", "", rule_2_mask
        )
        intermediate_stem_2 = replace_suffix(
            intermediate_stem_1, "ing", "", rule_3_mask
        )

        can_replace_mask = can_replace_mask & rule_2_or_rule_3_mask
        return apply_rule_list(
            intermediate_stem_2,
            [
                ("at", "ate", None),  # AT -> ATE
                ("bl", "ble", None),  # BL -> BLE
                ("iz", "ize", None),  # IZ -> IZE
                # (*d and not (*L or *S or *Z))
                # -> single letter
                (
                    "*d",
                    -1,  # intermediate_stem[-1],
                    lambda stem: last_char_not_in(
                        stem, characters=["l", "s", "z"]
                    ),
                ),
                # (m=1 and *o) -> E
                (
                    "",
                    "e",
                    lambda stem: measure_eq_n(stem, n=1) & ends_cvc(stem),
                ),
            ],
            can_replace_mask,
        )[0]

    def _step1c(self, word_str_ser, can_replace_mask=None):
        """Implements Step 1c from "An algorithm for suffix stripping"

        From the paper:

        Step 1c

            (*v*) Y -> I                    happy        ->  happi
                                            sky          ->  sky
        """
        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        def nltk_condition(stem):
            """
            This has been modified from the original Porter algorithm so
            that y->i is only done when y is preceded by a consonant,
            but not if the stem is only a single consonant, i.e.

            (*c and not c) Y -> I

            So 'happy' -> 'happi', but
            'enjoy' -> 'enjoy'  etc

            This is a much better rule. Formerly 'enjoy'->'enjoi' and
            'enjoyment'->'enjoy'. Step 1c is perhaps done too soon; but
            with this modification that no longer really matters.

            Also, the removal of the contains_vowel(z) condition means
            that 'spy', 'fly', 'try' ... stem to 'spi', 'fli', 'tri' and
            conflate with 'spied', 'tried', 'flies' ...
            """

            # equivalent to
            # len(stem) > 1 and self._is_consonant(stem, len(stem) - 1)
            len_gt_1_mask = len_gt_n(stem, 1)
            last_char_is_consonant_mask = is_consonant(stem, -1)
            return len_gt_1_mask & last_char_is_consonant_mask

        def original_condition(stem):
            return contains_vowel(stem)

        return apply_rule_list(
            word_str_ser,
            [
                (
                    "y",
                    "i",
                    nltk_condition
                    if self.mode == "NLTK_EXTENSIONS"
                    else original_condition,
                )
            ],
            can_replace_mask,
        )[0]

    def _step2(self, word_str_ser, can_replace_mask=None):
        """Implements Step 2 from "An algorithm for suffix stripping"

        From the paper:

        Step 2

            (m>0) ATIONAL ->  ATE       relational     ->  relate
            (m>0) TIONAL  ->  TION      conditional    ->  condition
                                        rational       ->  rational
            (m>0) ENCI    ->  ENCE      valenci        ->  valence
            (m>0) ANCI    ->  ANCE      hesitanci      ->  hesitance
            (m>0) IZER    ->  IZE       digitizer      ->  digitize
            (m>0) ABLI    ->  ABLE      conformabli    ->  conformable
            (m>0) ALLI    ->  AL        radicalli      ->  radical
            (m>0) ENTLI   ->  ENT       differentli    ->  different
            (m>0) ELI     ->  E         vileli        - >  vile
            (m>0) OUSLI   ->  OUS       analogousli    ->  analogous
            (m>0) IZATION ->  IZE       vietnamization ->  vietnamize
            (m>0) ATION   ->  ATE       predication    ->  predicate
            (m>0) ATOR    ->  ATE       operator       ->  operate
            (m>0) ALISM   ->  AL        feudalism      ->  feudal
            (m>0) IVENESS ->  IVE       decisiveness   ->  decisive
            (m>0) FULNESS ->  FUL       hopefulness    ->  hopeful
            (m>0) OUSNESS ->  OUS       callousness    ->  callous
            (m>0) ALITI   ->  AL        formaliti      ->  formal
            (m>0) IVITI   ->  IVE       sensitiviti    ->  sensitive
            (m>0) BILITI  ->  BLE       sensibiliti    ->  sensible
        """

        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        if self.mode == "NLTK_EXTENSIONS":
            # Instead of applying the ALLI -> AL rule after '(a)bli' per
            # the published algorithm, instead we apply it first, and,
            # if it succeeds, run the result through step2 again.

            alli_suffix_flag = ends_with_suffix(word_str_ser, "alli")
            stem_ser = replace_suffix(
                word_str_ser, "alli", "", alli_suffix_flag & can_replace_mask
            )
            positive_measure_flag = has_positive_measure(stem_ser)

            word_str_ser = replace_suffix(
                word_str_ser,
                "alli",
                "al",
                alli_suffix_flag & positive_measure_flag & can_replace_mask,
            )

        # not updating flag because nltk does not return

        bli_rule = ("bli", "ble", has_positive_measure)
        abli_rule = ("abli", "able", has_positive_measure)

        rules = [
            ("ational", "ate", has_positive_measure),
            ("tional", "tion", has_positive_measure),
            ("enci", "ence", has_positive_measure),
            ("anci", "ance", has_positive_measure),
            ("izer", "ize", has_positive_measure),
            abli_rule if self.mode == "ORIGINAL_ALGORITHM" else bli_rule,
            ("alli", "al", has_positive_measure),
            ("entli", "ent", has_positive_measure),
            ("eli", "e", has_positive_measure),
            ("ousli", "ous", has_positive_measure),
            ("ization", "ize", has_positive_measure),
            ("ation", "ate", has_positive_measure),
            ("ator", "ate", has_positive_measure),
            ("alism", "al", has_positive_measure),
            ("iveness", "ive", has_positive_measure),
            ("fulness", "ful", has_positive_measure),
            ("ousness", "ous", has_positive_measure),
            ("aliti", "al", has_positive_measure),
            ("iviti", "ive", has_positive_measure),
            ("biliti", "ble", has_positive_measure),
        ]

        if self.mode == "NLTK_EXTENSIONS":
            rules.append(("fulli", "ful", has_positive_measure))

            word_str_ser, can_replace_mask = apply_rule_list(
                word_str_ser, rules, can_replace_mask
            )

            # The 'l' of the 'logi' -> 'log' rule is put with the stem,
            # so that short stems like 'geo' 'theo' etc work like
            # 'archaeo' 'philo' etc.

            logi_suffix_flag = ends_with_suffix(word_str_ser, "logi")
            stem = word_str_ser.str.slice(stop=-3)
            measure_flag = has_positive_measure(stem)

            valid_flag = measure_flag & logi_suffix_flag & can_replace_mask
            return replace_suffix(word_str_ser, "logi", "log", valid_flag)

            # as below works on word rather than stem i don't
            # send it to apply rules but do it here
            # rules.append(
            # ("logi", "log", lambda stem:
            # self._has_positive_measure(word[:-3])
            # ))

        if self.mode == "MARTIN_EXTENSIONS":
            rules.append(("logi", "log", has_positive_measure))
            return apply_rule_list(word_str_ser, rules, can_replace_mask)[0]

    def _step3(self, word_str_ser, can_replace_mask=None):
        """Implements Step 3 from "An algorithm for suffix stripping"

        From the paper:

        Step 3

            (m>0) ICATE ->  IC              triplicate     ->  triplic
            (m>0) ATIVE ->                  formative      ->  form
            (m>0) ALIZE ->  AL              formalize      ->  formal
            (m>0) ICITI ->  IC              electriciti    ->  electric
            (m>0) ICAL  ->  IC              electrical     ->  electric
            (m>0) FUL   ->                  hopeful        ->  hope
            (m>0) NESS  ->                  goodness       ->  good
        """
        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        return apply_rule_list(
            word_str_ser,
            [
                ("icate", "ic", has_positive_measure),
                ("ative", "", has_positive_measure),
                ("alize", "al", has_positive_measure),
                ("iciti", "ic", has_positive_measure),
                ("ical", "ic", has_positive_measure),
                ("ful", "", has_positive_measure),
                ("ness", "", has_positive_measure),
            ],
            can_replace_mask,
        )[0]

    def _step4(self, word_str_ser, can_replace_mask=None):
        """Implements Step 4 from "An algorithm for suffix stripping"

        Step 4

            (m>1) AL    ->                  revival        ->  reviv
            (m>1) ANCE  ->                  allowance      ->  allow
            (m>1) ENCE  ->                  inference      ->  infer
            (m>1) ER    ->                  airliner       ->  airlin
            (m>1) IC    ->                  gyroscopic     ->  gyroscop
            (m>1) ABLE  ->                  adjustable     ->  adjust
            (m>1) IBLE  ->                  defensible     ->  defens
            (m>1) ANT   ->                  irritant       ->  irrit
            (m>1) EMENT ->                  replacement    ->  replac
            (m>1) MENT  ->                  adjustment     ->  adjust
            (m>1) ENT   ->                  dependent      ->  depend
            (m>1 and (*S or *T)) ION ->     adoption       ->  adopt
            (m>1) OU    ->                  homologou      ->  homolog
            (m>1) ISM   ->                  communism      ->  commun
            (m>1) ATE   ->                  activate       ->  activ
            (m>1) ITI   ->                  angulariti     ->  angular
            (m>1) OUS   ->                  homologous     ->  homolog
            (m>1) IVE   ->                  effective      ->  effect
            (m>1) IZE   ->                  bowdlerize     ->  bowdler

        The suffixes are now removed. All that remains is a little
        tidying up.
        """
        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )

        def measure_gt_1(ser):
            return measure_gt_n(ser, 1)

        return apply_rule_list(
            word_str_ser,
            [
                ("al", "", measure_gt_1),
                ("ance", "", measure_gt_1),
                ("ence", "", measure_gt_1),
                ("er", "", measure_gt_1),
                ("ic", "", measure_gt_1),
                ("able", "", measure_gt_1),
                ("ible", "", measure_gt_1),
                ("ant", "", measure_gt_1),
                ("ement", "", measure_gt_1),
                ("ment", "", measure_gt_1),
                ("ent", "", measure_gt_1),
                # (m>1 and (*S or *T)) ION ->
                (
                    "ion",
                    "",
                    lambda stem: measure_gt_n(stem, 1)
                    & last_char_in(stem, characters=["s", "t"]),
                ),
                ("ou", "", measure_gt_1),
                ("ism", "", measure_gt_1),
                ("ate", "", measure_gt_1),
                ("iti", "", measure_gt_1),
                ("ous", "", measure_gt_1),
                ("ive", "", measure_gt_1),
                ("ize", "", measure_gt_1),
            ],
            can_replace_mask,
        )[0]

    def _step5a(self, word_str_ser, can_replace_mask=None):
        """Implements Step 5a from "An algorithm for suffix stripping"

        From the paper:

        Step 5a

            (m>1) E     ->                  probate        ->  probat
                                            rate           ->  rate
            (m=1 and not *o) E ->           cease          ->  ceas
        """

        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )
        # Note that Martin's test vocabulary and reference
        # implementations are inconsistent in how they handle the case
        # where two rules both refer to a suffix that matches the word
        # to be stemmed, but only the condition of the second one is
        # true.
        # Earlier in step2b we had the rules:
        #     (m>0) EED -> EE
        #     (*v*) ED  ->
        # but the examples in the paper included "feed"->"feed", even
        # though (*v*) is true for "fe" and therefore the second rule
        # alone would map "feed"->"fe".
        # However, in THIS case, we need to handle the consecutive rules
        # differently and try both conditions (obviously; the second
        # rule here would be redundant otherwise). Martin's paper makes
        # no explicit mention of the inconsistency; you have to infer it
        # from the examples.
        # For this reason, we can't use _apply_rule_list here.

        ##

        # logic is equivalent to below
        # if word.endswith('e'):
        #  stem = self._replace_suffix(word, 'e', '')
        #  if self._measure(stem) > 1:
        #      return stem  rule_1
        #  if self._measure(stem) == 1 and not self._ends_cvc(stem):
        #      return stem  rule_2
        #

        e_suffix_flag = ends_with_suffix(word_str_ser, "e")
        stem = replace_suffix(
            word_str_ser, "e", "", e_suffix_flag & can_replace_mask
        )

        measure_gt_1_flag = measure_gt_n(stem, 1)

        # if self._measure(stem) > 1:
        rule_1_flag = measure_gt_1_flag

        # if measure==1 and not self._ends_cvc(stem):
        measure_eq_1_flag = measure_eq_n(stem, 1)
        does_not_ends_with_cvc_flag = ~ends_cvc(stem)
        rule_2_flag = measure_eq_1_flag & does_not_ends_with_cvc_flag

        overall_rule_flag = (
            (rule_1_flag | rule_2_flag) & e_suffix_flag & can_replace_mask
        )

        return replace_suffix(word_str_ser, "e", "", overall_rule_flag)

    def _step5b(self, word_str_ser, can_replace_mask=None):
        """Implements Step 5a from "An algorithm for suffix stripping"

        From the paper:

        Step 5b

            (m > 1 and *d and *L) -> single letter
                                    controll       ->  control
                                    roll           ->  roll
        """

        can_replace_mask = build_can_replace_mask(
            len_mask=len(word_str_ser), mask=can_replace_mask
        )
        # word, [('ll', 'l', lambda stem: self._measure(word[:-1]) > 1)]
        # because here we are applying rule on word instead of stem
        # so, unlike nltk we don't use apply rules

        ll_suffix_flag = ends_with_suffix(word_str_ser, "ll")

        stem = word_str_ser.str.slice()
        measure_gt_1_flag = measure_gt_n(stem, 1)

        valid_flag = measure_gt_1_flag & ll_suffix_flag & can_replace_mask

        return replace_suffix(word_str_ser, "ll", "l", valid_flag)


def map_irregular_forms(word_str_ser, can_replace_mask):
    # replaces all strings and stop rules
    # need to process it
    irregular_forms = {
        "sky": ["sky", "skies"],
        "die": ["dying"],
        "lie": ["lying"],
        "tie": ["tying"],
        "news": ["news"],
        "inning": ["innings", "inning"],
        "outing": ["outings", "outing"],
        "canning": ["cannings", "canning"],
        "howe": ["howe"],
        "proceed": ["proceed"],
        "exceed": ["exceed"],
        "succeed": ["succeed"],
    }
    for replacement, form_ls in irregular_forms.items():
        for form in form_ls:
            equal_flag = word_str_ser == form
            stem_ser = get_stem_series(
                word_str_ser, len(form), can_replace_mask & equal_flag
            )
            replacement_ser = get_str_replacement_series(
                replacement, can_replace_mask & equal_flag
            )

            word_str_ser = stem_ser.str.cat(replacement_ser)
            can_replace_mask &= ~equal_flag

    return word_str_ser, can_replace_mask


def get_condition_flag(word_str_ser, condition):
    """
    condition  = None or a function that returns a bool series
    return a bool series where flag is valid
    """
    if condition is None:
        return cudf.Series(cp.ones(len(word_str_ser), bool))
    else:
        return condition(word_str_ser)


def apply_rule(word_str_ser, rule, w_in_c_flag):
    """Applies the first applicable suffix-removal rule to the word

    Takes a word and a list of suffix-removal rules represented as
    3-tuples, with the first element being the suffix to remove,
    the second element being the string to replace it with, and the
    final element being the condition for the rule to be applicable,
    or None if the rule is unconditional.
    """
    suffix, replacement, condition = rule
    if suffix == "*d":
        double_consonant_mask = ends_with_double_constant(word_str_ser)
        # all flags needed  here
        # with **d in nltk we pass word_series rather than stem_series
        # see below:
        # lambda stem: intermediate_stem[-1] not in ('l', 's', 'z'),
        # condition is on  intermediate_stem
        intermediate_stem = word_str_ser.str.slice(stop=-1)
        condition_mask = get_condition_flag(intermediate_stem, condition)

        # mask where replacement will happen
        valid_mask = double_consonant_mask & condition_mask & w_in_c_flag

        # recent cuDF change made it so that the conditions above have a NA
        # instead of null, which makes us need to replace them with False
        # here so replace_suffix works correctly and doesn't duplicate
        # single letters we don't want to.
        valid_mask = valid_mask.fillna(False)

        # new series with updated valid_mask
        word_str_ser = replace_suffix(
            word_str_ser, suffix, replacement, valid_mask
        )
        w_in_c_flag &= ~double_consonant_mask

    else:

        suffix_mask = ends_with_suffix(word_str_ser, suffix)
        valid_mask = suffix_mask & w_in_c_flag

        stem_ser = replace_suffix(word_str_ser, suffix, "", valid_mask)

        condition_mask = get_condition_flag(stem_ser, condition)
        # mask where replacement will happen
        valid_mask = condition_mask & suffix_mask & w_in_c_flag
        word_str_ser = replace_suffix(
            word_str_ser, suffix, replacement, valid_mask
        )

        # we wont apply further rules if it has a matching suffix
        w_in_c_flag &= ~suffix_mask

    return word_str_ser, w_in_c_flag


def apply_rule_list(word_str_ser, rules, condition_flag):
    """Applies the first applicable suffix-removal rule to the word

    Takes a word series and a list of suffix-removal rules represented as
    3-tuples, with the first element being the suffix to remove,
    the second element being the string to replace it with, and the
    final element being the condition for the rule to be applicable,
    or None if the rule is unconditional.
    """

    for rule in rules:
        word_str_ser, condition_flag = apply_rule(
            word_str_ser, rule, condition_flag
        )

    return word_str_ser, condition_flag


def build_can_replace_mask(len_mask, mask):
    """
    Creates a cudf series representing can_replace_mask of length=len_mask
    if mask is None else returns mask
    """
    if mask is None:
        mask = cudf.Series(cp.ones(len_mask, dtype=bool))
    return mask
