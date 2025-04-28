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

import numpy as np
import pytest

from cuml.internals.input_utils import input_to_cuml_array
from cuml.tsa import auto_arima

###############################################################################
#                       Helpers and reference functions                       #
###############################################################################


def _build_division_map_ref(id_tracker, batch_size, n_sub):
    """Reference implementation for _build_division_map in pure Python"""
    id_to_model = np.zeros(batch_size, dtype=np.int32)
    id_to_pos = np.zeros(batch_size, dtype=np.int32)
    for i in range(n_sub):
        id_to_model[id_tracker[i]] = i
        for j in range(len(id_tracker[i])):
            id_to_pos[id_tracker[i][j]] = j
    return id_to_model, id_to_pos


###############################################################################
#                                    Tests                                    #
###############################################################################


@pytest.mark.parametrize("batch_size", [10, 100])
@pytest.mark.parametrize("n_obs", [31, 65])
@pytest.mark.parametrize("prop_true", [0, 0.5, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_divide_by_mask(batch_size, n_obs, prop_true, dtype):
    """Test the helper that splits a dataset in 2 based on a boolean mask"""
    # Generate random data, mask and batch indices
    data_np = (
        (np.random.uniform(-1.0, 1.0, (batch_size, n_obs)))
        .astype(dtype)
        .transpose()
    )
    nb_true = int(prop_true * batch_size)
    mask_np = np.random.permutation(
        [False] * (batch_size - nb_true) + [True] * nb_true
    )
    b_id_np = np.array(range(batch_size), dtype=np.int32)
    data, *_ = input_to_cuml_array(data_np)
    mask, *_ = input_to_cuml_array(mask_np)
    b_id, *_ = input_to_cuml_array(b_id_np)

    # Call the tested function
    sub_data, sub_id = [None, None], [None, None]
    (
        sub_data[0],
        sub_id[0],
        sub_data[1],
        sub_id[1],
    ) = auto_arima._divide_by_mask(data, mask, b_id)

    # Compute the expected results in pure Python
    sub_data_ref = [data_np[:, np.logical_not(mask_np)], data_np[:, mask_np]]
    sub_id_ref = [b_id_np[np.logical_not(mask_np)], b_id_np[mask_np]]

    # Compare the results
    for i in range(2):
        # First check the cases of empty sub-batches
        if sub_data[i] is None:
            # The reference must be empty
            assert sub_data_ref[i].shape[1] == 0
            # And the id array must be None too
            assert sub_id[i] is None
        # When the sub-batch is not empty, compare to the reference
        else:
            np.testing.assert_allclose(
                sub_data[i].to_output("numpy"), sub_data_ref[i]
            )
            np.testing.assert_array_equal(
                sub_id[i].to_output("numpy"), sub_id_ref[i]
            )


@pytest.mark.parametrize("batch_size", [10, 100])
@pytest.mark.parametrize("n_obs", [31, 65])
@pytest.mark.parametrize("n_sub", [1, 2, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_divide_by_min(batch_size, n_obs, n_sub, dtype):
    """Test the helper that splits a dataset by selecting the minimum
    of a given criterion
    """
    # Generate random data, metrics and batch indices
    data_np = (
        (np.random.uniform(-1.0, 1.0, (batch_size, n_obs)))
        .astype(dtype)
        .transpose()
    )
    crit_np = (
        (np.random.uniform(-1.0, 1.0, (n_sub, batch_size)))
        .astype(dtype)
        .transpose()
    )
    b_id_np = np.array(range(batch_size), dtype=np.int32)
    data, *_ = input_to_cuml_array(data_np)
    crit, *_ = input_to_cuml_array(crit_np)
    b_id, *_ = input_to_cuml_array(b_id_np)

    # Call the tested function
    sub_batches, sub_id = auto_arima._divide_by_min(data, crit, b_id)

    # Compute the expected results in pure Python
    which_sub = crit_np.argmin(axis=1)
    sub_batches_ref = []
    sub_id_ref = []
    for i in range(n_sub):
        sub_batches_ref.append(data_np[:, which_sub == i])
        sub_id_ref.append(b_id_np[which_sub == i])

    # Compare the results
    for i in range(n_sub):
        # First check the cases of empty sub-batches
        if sub_batches[i] is None:
            # The reference must be empty
            assert sub_batches_ref[i].shape[1] == 0
            # And the id array must be None too
            assert sub_id[i] is None
        # When the sub-batch is not empty, compare to the reference
        else:
            np.testing.assert_allclose(
                sub_batches[i].to_output("numpy"), sub_batches_ref[i]
            )
            np.testing.assert_array_equal(
                sub_id[i].to_output("numpy"), sub_id_ref[i]
            )


@pytest.mark.parametrize("batch_size", [25, 103, 1001])
@pytest.mark.parametrize("n_sub", [1, 2, 10])
def test_build_division_map(batch_size, n_sub):
    """Test the helper that builds a map of the new sub-batch and position
    in this batch of each series in a divided batch
    """
    # Generate the id tracker
    # Note: in the real use case the individual id arrays are sorted but the
    # helper function doesn't require that
    tracker_np = np.array_split(np.random.permutation(batch_size), n_sub)
    tracker = [
        input_to_cuml_array(tr, convert_to_dtype=np.int32)[0]
        for tr in tracker_np
    ]

    # Call the tested function
    id_to_model, id_to_pos = auto_arima._build_division_map(
        tracker, batch_size
    )

    # Compute the expected results in pure Python
    id_to_model_ref, id_to_pos_ref = _build_division_map_ref(
        tracker_np, batch_size, n_sub
    )

    # Compare the results
    np.testing.assert_array_equal(
        id_to_model.to_output("numpy"), id_to_model_ref
    )
    np.testing.assert_array_equal(id_to_pos.to_output("numpy"), id_to_pos_ref)


@pytest.mark.parametrize("batch_size", [10, 100])
@pytest.mark.parametrize("n_obs", [31, 65])
@pytest.mark.parametrize("n_sub", [1, 2, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_merge_series(batch_size, n_obs, n_sub, dtype):
    """Test the helper that merges a divided batch based on division maps
    that track the sub-batch and position of each member
    """
    # Generate an id tracker and compute id_to_sub and id_to_pos
    tracker_np = np.array_split(np.random.permutation(batch_size), n_sub)
    id_to_sub_np, id_to_pos_np = _build_division_map_ref(
        tracker_np, batch_size, n_sub
    )
    id_to_sub, *_ = input_to_cuml_array(
        id_to_sub_np, convert_to_dtype=np.int32
    )
    id_to_pos, *_ = input_to_cuml_array(
        id_to_pos_np, convert_to_dtype=np.int32
    )

    # Generate the final dataset (expected result)
    data_np = (
        (np.random.uniform(-1.0, 1.0, (batch_size, n_obs)))
        .astype(dtype)
        .transpose()
    )

    # Divide the dataset according to the id tracker
    data_div = []
    for i in range(n_sub):
        data_piece = np.zeros(
            (n_obs, len(tracker_np[i])), dtype=dtype, order="F"
        )
        for j in range(len(tracker_np[i])):
            data_piece[:, j] = data_np[:, tracker_np[i][j]]
        data_div.append(input_to_cuml_array(data_piece)[0])

    # Call the tested function
    data = auto_arima._merge_series(data_div, id_to_sub, id_to_pos, batch_size)

    # Compare the results
    np.testing.assert_allclose(data.to_output("numpy"), data_np)
