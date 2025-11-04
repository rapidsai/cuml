#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
from nltk import stem as nltk_stem

from cuml.preprocessing.text import stem as rapids_stem


def test_same_results():
    # This is a list of 100 random words from nltk treebank.
    word_ls = [
        "marie-louise",
        "acid",
        "107.03",
        "private",
        "focus",
        "refocusing",
        "*-56",
        "moscow",
        "skipped",
        "orchestrated",
        "asian",
        "car-care",
        "fiber",
        "sensation",
        "stock-specialist",
        "china",
        "19.95",
        "agreeing",
        "shudders",
        "bone",
        "5.5",
        "23,000",
        "modification",
        "afraid",
        "anytime",
        "bidders",
        "breathe",
        "year-long",
        "36.9",
        "mariotta",
        "migrate",
        "8.64",
        "shelby",
        "tray",
        "rogers",
        "tarwhine",
        "227",
        "louisville",
        "enterprise",
        "cs",
        "2.30",
        "vagrant",
        "filmed",
        "eggers",
        "boosters",
        "yen-support",
        "60",
        "37",
        "warehouses",
        "likely",
        "machine-gun-toting",
        "evaluation",
        "2",
        "west",
        "contracts",
        "withdraw",
        "nine-month",
        "attempting",
        "seats",
        "insiders",
        "17.95",
        "neal",
        "laura",
        "25.6",
        "anti-abortionists",
        "melt-textured",
        "slides",
        "milne",
        "fall",
        "means",
        "biscayne",
        "taxpayer",
        "ban",
        "awareness",
        "mississippi",
        "oppose",
        "default",
        "michigan",
        "severable",
        "souper",
        "requirements",
        "media",
        "preferred",
        "write",
        "kit",
        "seller",
        "corrigan",
        "hysteria",
        "elisa",
        "british",
        "earns",
        "traficant",
        "surviving",
        "home",
        "indicator",
        "bread-and-butter",
        "undertone",
        "dec.",
        "nomura",
        "floor",
    ]
    word_ser = cudf.Series(word_ls)

    nltk_stemmer = nltk_stem.PorterStemmer()
    nltk_stemmed = [nltk_stemmer.stem(word) for word in word_ls]

    cuml_stemmer = rapids_stem.PorterStemmer()
    cuml_stemmed = cuml_stemmer.stem(word_ser)

    assert all(
        [a == b for a, b in zip(nltk_stemmed, cuml_stemmed.to_pandas().values)]
    )
