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

#ifdef WITH_UCX
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
static const ucp_tag_t default_tag_mask = -1;

/**
 * @brief Continues progress on worker until a message request has completed
 */
static void wait(ucp_worker_h ucp_worker, struct ucx_context *context) {
  while (context->completed == 0) {
    ucp_worker_progress(ucp_worker);
  }
}

/**
 * @brief callback for flushing worker
 */
static void flush_callback(void *request, ucs_status_t status) {}

/**
 * @brief Flush the send buffer on the given endpoint for the given worker
 */
static ucs_status_t flush_ep(ucp_worker_h worker, ucp_ep_h ep) {
  void *request;

  request = ucp_ep_flush_nb(ep, 0, flush_callback);
  if (request == NULL) {
    return UCS_OK;
  } else if (UCS_PTR_IS_ERR(request)) {
    return UCS_PTR_STATUS(request);
  } else {
    ucs_status_t status;
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
    ucp_request_release(request);
    return status;
  }
}

/**
 * @brief Asynchronous send callback sets request to completed
 */
static void send_handle(void *request, ucs_status_t status) {
  struct ucx_context *context = (struct ucx_context *)request;
  context->completed = 1;

  printf("[0x%x] send handler called with status %d (%s)\n",
         (unsigned int)pthread_self(), status, ucs_status_string(status));
}

/**
 * @brief Asynchronous recv callback sets request to completed
 */
static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info) {
  struct ucx_context *context = (struct ucx_context *)request;
  context->completed = 1;

  printf("[0x%x] receive handler called with status %d (%s), length %lu\n",
         (unsigned int)pthread_self(), status, ucs_status_string(status),
         info->length);
}

/**
 * @brief Asynchronously send data to the given endpoint using the given tag
 */
struct ucx_context *ucp_isend(ucp_ep_h ep_ptr, const void *buf, int size,
                              int tag) {
  struct ucx_context *ucp_request = 0;
  ucp_tag_t ucp_tag = (ucp_tag_t)tag;

  ucp_request = (struct ucx_context *)ucp_tag_send_nb(
    ep_ptr, buf, size, ucp_dt_make_contig(1), ucp_tag, send_handle);

  /**
   * On error, close endpoint, set request to completed and close
   * @TODO: Should the request be closed here as well? How should we handle this?
   * @TODO: Should the Python layer be attempting to reconnect endpoints when they are closed?
   */
  if (UCS_PTR_IS_ERR(ucp_request)) {
    printf("unable to send UCX data message\n");
    ucp_ep_close_nb(ep_ptr, UCP_EP_CLOSE_MODE_FLUSH);
    return nullptr;
  /**
   * If the request didn't fail, but it's not OK, it is in flight.
   * Expect the handler to be invoked
   */
  } else if (UCS_PTR_STATUS(ucp_request) != UCS_OK) {
    printf("Message is sending. Handler should be invoked.\n");

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
struct ucx_context *ucp_irecv(ucp_worker_h worker, ucp_ep_h ep_ptr, void *buf,
                              int size, int tag) {
  ucp_tag_t ucp_tag = (ucp_tag_t)tag;

  struct ucx_context *ucp_request = (struct ucx_context *)ucp_tag_recv_nb(
    worker, buf, size, ucp_dt_make_contig(1), ucp_tag, default_tag_mask,
    recv_handle);

  /**
   * If error, endpoint is closed.
   * @TODO: Should the request be closed here as well? How should we handle this?
   * @TODO: Should the Python layer be attempting to reconnect endpoints when they are closed?
   */
  if (UCS_PTR_IS_ERR(ucp_request)) {
    printf("unable to receive UCX data message (%d)\n");
    //       UCS_PTR_STATUS(ucp_request));
    ucp_ep_close_nb(ep_ptr, UCP_EP_CLOSE_MODE_FLUSH);
    return nullptr;

  /**
   * Otherwise, request is successful and handler should get invoked on it.
   */
  } else {

    //wait(_ucp_worker, ucp_request);
    //ucp_request->completed = 0;
    //ucp_request_release(request);
  }

  return ucp_request;
}
#endif
