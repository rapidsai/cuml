/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include <mpi.h>

#include <cuML.hpp>
#include <cuML_comms.hpp>
#include <sandbox/test.hpp>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

int main(int argc, char * argv[])
{
    MPI_CALL(MPI_Init(&argc, &argv));

    MPI_Comm cuml_mpi_comm;
    MPI_CALL(MPI_Comm_dup(MPI_COMM_WORLD, &cuml_mpi_comm));

    {
        ML::cumlHandle cumlHandle;
        initialize_mpi_comms(cumlHandle, cuml_mpi_comm);
        ML::sandbox::mpi_test(cumlHandle);
    }

    MPI_CALL(MPI_Comm_free(&cuml_mpi_comm));

    MPI_CALL(MPI_Finalize());
    return 0;
}
