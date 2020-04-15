# Not a real test, just demo
import pytest
import json
import numpy as np

def test_model(golden):
    if golden.recompute:
        golden.accuracy_ = 1.0

    new_value = 1.0 # We would actually do some compute here
    assert golden.accuracy_ == new_value

@pytest.mark.parametrize('sample_value', [1, 2])
def test_other_model(golden, sample_value):
    if golden.recompute:
        print("Recomputing test_failure")
        golden.accuracy_ = 0.99 + sample_value

    assert golden.accuracy_ == 0.99 + sample_value

@pytest.mark.xfail(strict=False)
def test_failure(golden):
    if golden.recompute:
        print("Recomputing test_failure")
        golden.some_metric_ = 0.50

    assert golden.some_metric_ == 0.98

def test_numpy_arrays(golden):
    if golden.recompute:
        golden.good_values_ = np.array([0.0, 1.0, 2.0]).tolist()

    assert np.array_equal(np.array(golden.good_values_), np.array([0.0, 1.0, 2.0]))
