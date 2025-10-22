#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

EXITCODE=0

for nb in "$@"; do
    NBFILENAME=$nb
    shift

    echo --------------------------------------------------------------------------------
    echo STARTING: "${NBFILENAME}"
    echo --------------------------------------------------------------------------------
    time bash -c "jupyter execute ${NBFILENAME}; EC=\$?; echo -------------------------------------------------------------------------------- ; echo DONE: ${NBFILENAME}; exit \$EC"
    NBEXITCODE=$?
    echo EXIT CODE: ${NBEXITCODE}
    echo
    EXITCODE=$((EXITCODE | "${NBEXITCODE}"))
done

exit ${EXITCODE}
