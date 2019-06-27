
#include <cuML.hpp>
#include <cuML_comms.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>

#include <ucp/api/ucp.h>

#include <stdio.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

namespace ML {

  void inject_comms_py(ML::cumlHandle *handle, ncclComm_t comm, void *ucp_worker, void *eps, int size, int rank) {
    ucp_worker_print_info((ucp_worker_h)ucp_worker, stdout);

    ucp_ep_h *new_ep_arr = new ucp_ep_h[size];

    size_t *size_t_ep_arr = (size_t*)eps;

    for(int i = 0; i < size; i++) {

        size_t ptr = size_t_ep_arr[i];
        if(ptr != 0) {
            ucp_ep_h *eps_ptr = (ucp_ep_h*)size_t_ep_arr[i];
            new_ep_arr[i] = *eps_ptr;
        } else {
            new_ep_arr[i] = nullptr;
        }
    }

    inject_comms(*handle, comm, (ucp_worker_h)ucp_worker, (ucp_ep_h*) new_ep_arr, size, rank);

  }

  void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId) {
      memcpy(id->internal, uniqueId, NCCL_UNIQUE_ID_BYTES);
  }

  /**
   * @brief Returns a NCCL unique ID as a character array. PyTorch
   * uses this same approach, so that it can be more easily
   * converted to a native Python string by Cython and further
   * serialized to be sent across process & node boundaries.
   *
   * @returns the generated NCCL unique ID for establishing a
   * new clique.
   */
  void get_unique_id(char *uid) {

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    memcpy(uid, id.internal, NCCL_UNIQUE_ID_BYTES);
  }
}
