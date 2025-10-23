#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef str _get_treelite_error():
    cdef str err = TreeliteGetLastError().decode("UTF-8")
    return err


def safe_treelite_call(res: int, err_msg: str) -> None:
    if res < 0:
        raise RuntimeError(f"{err_msg}\n{_get_treelite_error()}")
