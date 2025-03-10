#
# Copyright (c) 2025, NVIDIA CORPORATION.
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


def cudf_to_pandas(cudf_obj):
    """Convert a ``cudf`` object to a ``pandas`` (or ``cudf.pandas``) object.

    Unlike cudf's builtin ``to_pandas`` method, this function will return a
    ``cudf.pandas`` object if ``cudf.pandas`` is active.
    """
    import cudf.pandas

    if cudf.pandas.LOADED:
        return cudf.pandas.as_proxy_object(cudf_obj)
    else:
        return cudf_obj.to_pandas()


def output_to_df_obj_like(X_out, X_in, output_type):
    """Cast CumlArray `X_out` to the dataframe / series type as `X_in`
    `CumlArray` abstracts away the dataframe / series metadata, when API
    methods needs to return a dataframe / series matching original input
    metadata, this function can copy input metadata to output.
    """

    if output_type not in ["series", "dataframe"]:
        raise ValueError(
            f'output_type must be either "series" or "dataframe" : {output_type}'
        )

    out = None
    if output_type == "series":
        out = X_out.to_output("series")
        out.name = X_in.name
    elif output_type == "dataframe":
        out = X_out.to_output("dataframe")
        out.columns = X_in.columns
    return out
