import cudf
from cuml.preprocessing.text.stem.porter_stemmer import PorterStemmer


def test_step1a():
    word_strs = cudf.Series(["caresses", "ponies", "ties", "caress", "cats"])

    st = PorterStemmer()
    got = st._step1a(word_strs)

    expect = ["caress", "poni", "tie", "caress", "cat"]
    assert list(got.to_pandas().values) == expect


def test_step1b():
    word_strs_ls = [
        "feed",
        "agreed",
        "plastered",
        "bled",
        "motoring",
        "sing",
        "conflated",
        "troubled",
        "sized",
        "hopping",
        "tanned",
        "falling",
        "hissing",
        "fizzed",
        "failing",
        "filing",
    ]

    expected = [
        "feed",
        "agree",
        "plaster",
        "bled",
        "motor",
        "sing",
        "conflate",
        "trouble",
        "size",
        "hop",
        "tan",
        "fall",
        "hiss",
        "fizz",
        "fail",
        "file",
    ]

    word_strs = cudf.Series(word_strs_ls)
    st = PorterStemmer()
    got = st._step1b(word_strs)

    assert list(got.to_pandas().values) == expected


def test_step1c():
    word_strs_ls = ["happy", "sky", "enjoy", "boy", "toy", "y"]
    word_strs = cudf.Series(word_strs_ls)
    st = PorterStemmer()
    got = st._step1c(word_strs)

    expect = ["happi", "ski", "enjoy", "boy", "toy", "y"]
    assert list(got.to_pandas().values) == expect


def test_step2():
    word_strs_ls = [
        "relational",
        "conditional",
        "rational",
        "valenci",
        "hesitanci",
        "digitizer",
        "conformabli",
        "radicalli",
        "differentli",
        "vileli",
        "analogousli",
        "vietnamization",
        "predication",
        "operator",
        "feudalism",
        "decisiveness",
        "hopefulness",
        "callousness",
        "formaliti",
        "sensitiviti",
        "sensibiliti",
    ]

    expect = [
        "relate",
        "condition",
        "rational",
        "valence",
        "hesitance",
        "digitize",
        "conformable",
        "radical",
        "different",
        "vile",
        "analogous",
        "vietnamize",
        "predicate",
        "operate",
        "feudal",
        "decisive",
        "hopeful",
        "callous",
        "formal",
        "sensitive",
        "sensible",
    ]

    word_strs = cudf.Series(word_strs_ls)
    st = PorterStemmer()
    got = st._step2(word_strs)
    assert list(got.to_pandas().values) == expect


def test_step3():
    word_strs_ls = [
        "triplicate",
        "formative",
        "formalize",
        "electriciti",
        "electriciti",
        "hopeful",
        "goodness",
    ]
    expect = ["triplic", "form", "formal", "electric", "electric", "hope", "good"]

    word_strs = cudf.Series(word_strs_ls)
    st = PorterStemmer()
    got = st._step3(word_strs)
    assert list(got.to_pandas().values) == expect


def test_step4():
    word_strs_ls = [
        "revival",
        "allowance",
        "inference",
        "airliner",
        "gyroscopic",
        "adjustable",
        "defensible",
        "irritant",
        "replacement",
        "adjustment",
        "dependent",
        "adoption",
        "homologou",
        "communism",
        "activate",
        "angulariti",
        "homologous",
        "effective",
        "bowdlerize",
    ]

    expect = [
        "reviv",
        "allow",
        "infer",
        "airlin",
        "gyroscop",
        "adjust",
        "defens",
        "irrit",
        "replac",
        "adjust",
        "depend",
        "adopt",
        "homolog",
        "commun",
        "activ",
        "angular",
        "homolog",
        "effect",
        "bowdler",
    ]

    word_strs = cudf.Series(word_strs_ls)
    st = PorterStemmer()
    got = st._step4(word_strs)
    assert list(got.to_pandas().values) == expect


def test_step5a():
    word_strs_ls = ["probate", "rate", "cease", "ones"]
    word_strs = cudf.Series(word_strs_ls)

    expect = ["probat", "rate", "ceas", "ones"]
    st = PorterStemmer()
    got = st._step5a(word_strs)
    assert list(got.to_pandas().values) == expect


def test_step5b():
    word_strs_ls = ["controll", "roll"]
    word_strs = cudf.Series(word_strs_ls)
    expect = ["control", "roll"]

    st = PorterStemmer()
    got = st._step5b(word_strs)
    assert list(got.to_pandas().values) == expect
