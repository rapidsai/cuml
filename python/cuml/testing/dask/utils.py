# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from sklearn.datasets import fetch_20newsgroups
from cuml.dask.common import to_sparse_dask_array
from sklearn.feature_extraction.text import HashingVectorizer
import dask
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")


def load_text_corpus(client):

    categories = [
        "alt.atheism",
        "soc.religion.christian",
        "comp.graphics",
        "sci.med",
    ]
    twenty_train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )

    hv = HashingVectorizer(alternate_sign=False, norm=None)

    xformed = hv.fit_transform(twenty_train.data).astype(cp.float32)

    X = to_sparse_dask_array(xformed, client)

    y = dask.array.from_array(
        twenty_train.target, asarray=False, fancy=False
    ).astype(cp.int32)

    return X, y
