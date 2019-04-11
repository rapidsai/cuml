# Copyright (c) 2019, NVIDIA CORPORATION.
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


def split(array, lengths):
    start_idx = 0
    splitted = []
    for i in range(lengths.shape[0]):
        end_idx = start_idx + lengths[i]
        splitted.append(array[start_idx:end_idx])
        start_idx += lengths[i]
    return splitted


# def compare_state_seq(seq_x, seq_y, lengths):
#     def compare(seq_a, seq_y) :
#
#         # Create mapping
#
#         # Compare
#         same = True
#         # for x, y in zip(seq_x, seq_y):
#         #     if x
#
#     len = seq_x.shape[0]
#     nSeq = lengths.shape[0]
#
#     splitted_x = split(seq_x, lengths)
#     splitted_y = split(seq_y, lengths)
