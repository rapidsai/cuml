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

#include <dlfcn.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>

#include <utils.h>

static const ucp_tag_t default_tag_mask = -1;

static const ucp_tag_t any_rank_tag_mask = 0x0000FFFF;

static const int UCP_ANY_RANK = -1;

/**
 * @brief Asynchronous send callback sets request to completed
 */
static void send_handle(void *request, ucs_status_t status) {
  struct ucx_context *context = (struct ucx_context *)request;
  context->completed = 1;
}

/**
 * @brief Asynchronous recv callback sets request to completed
 */
static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info) {
  struct ucx_context *context = (struct ucx_context *)request;
  context->completed = 1;
}

void load_ucp_handle(void *handle) {
  handle = dlopen("libucp.so", RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
  if (!handle) {
    handle = dlopen("libucp.so", RTLD_LAZY | RTLD_NODELETE);
    if (!handle) {
      fprintf(stderr, "Cannot open UCX library: %s\n", dlerror());
      exit(1);
    }
  }
  dlerror();
}

void close_ucp_handle(void *handle) { dlclose(handle); }

/**
 * @brief Asynchronously send data to the given endpoint using the given tag
 */
struct ucx_context *ucp_isend(void *ucp_handle, ucp_ep_h ep_ptr,
                              const void *buf, int size, int tag,
                              ucp_tag_t tag_mask, int rank) {
  ucp_tag_t ucp_tag = ((uint32_t)rank << 31) | (uint32_t)tag;

  ucs_status_t (*send_func)(ucp_ep_h ep, const void *buffer, size_t count,
                            ucp_datatype_t datatype, ucp_tag_t tag,
                            ucp_send_callback_t cb) =
    *(ucs_status_t(*)(ucp_ep_h ep, const void *buffer, size_t count,
                      ucp_datatype_t datatype, ucp_tag_t tag,
                      ucp_send_callback_t cb))dlsym(ucp_handle,
                                                    "ucp_tag_send_nb");

  char *error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "Error loading function symbol: %s\n", error);
    exit(1);
  }

  struct ucx_context *ucp_request = (struct ucx_context *)ucp_tag_send_nb(
    ep_ptr, buf, size, ucp_dt_make_contig(1), ucp_tag, send_handle);

  if (UCS_PTR_IS_ERR(ucp_request)) {
    ASSERT(!UCS_PTR_IS_ERR(ucp_request),
           "unable to send UCX data message (%d)\n",
           UCS_PTR_STATUS(ucp_request));
    /**
   * If the request didn't fail, but it's not OK, it is in flight.
   * Expect the handler to be invoked
   */
  } else if (UCS_PTR_STATUS(ucp_request) != UCS_OK) {
    /**
    * If the request is OK, it's already been completed and we don't need to wait on it.
    * The request will be a nullptr, however, so we need to create a new request
    * and set it to completed to make the "waitall()" function work properly.
    */
  } else {
    ucp_request = (struct ucx_context *)malloc(sizeof(struct ucx_context));
    ucp_request->completed = 1;
    ucp_request->needs_release = false;
  }

  return ucp_request;
}

/**
 * @bried Asynchronously receive data from given endpoint with the given tag.
 */
struct ucx_context *ucp_irecv(void *ucp_handle, ucp_worker_h worker,
                              ucp_ep_h ep_ptr, void *buf, int size, int tag,
                              ucp_tag_t tag_mask, int sender_rank) {
  ucp_tag_t ucp_tag = ((uint32_t)sender_rank << 31) | (uint32_t)tag;

  ucs_status_t (*recv_func)(ucp_worker_h worker, void *buffer, size_t count,
                            ucp_datatype_t datatype, ucp_tag_t tag,
                            ucp_tag_t tag_mask, ucp_tag_recv_callback_t cb) =
    (ucs_status_t(*)(ucp_worker_h worker, void *buffer, size_t count,
                     ucp_datatype_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                     ucp_tag_recv_callback_t cb))dlsym(ucp_handle,
                                                       "ucp_tag_recv_nb");

  char *error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "Error loading function symbol: %s\n", error);
    exit(1);
  }

  struct ucx_context *ucp_request = (struct ucx_context *)recv_func(
    worker, buf, size, ucp_dt_make_contig(1), ucp_tag, tag_mask, recv_handle);

  if (UCS_PTR_IS_ERR(ucp_request)) {
    ASSERT(!UCS_PTR_IS_ERR(ucp_request),
           "unable to receive UCX data message (%d)\n",
           UCS_PTR_STATUS(ucp_request));
  }

  return ucp_request;
}
