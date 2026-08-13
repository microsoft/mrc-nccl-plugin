/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2015-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

#include "ibvwrap.h"
#include "utils.h"
#include "nccl.h"
#include "param.h"
#include "mrc.h"


#define IBV_PTR_CHECK_ERRNO(call, retval, error_retval, name) \
  retval = call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PTR_CHECK(call, retval, error_retval, name) \
  retval = call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO_OPTIONAL(call, success_retval, name, supported) \
  int ret = call; \
  if (ret == ENOTSUP || ret == EOPNOTSUPP) { \
    INFO(NCCL_NET, "Call to " name " not supported"); \
    *supported = 0; \
    return ncclSuccess; \
  } else if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    *supported = 1; \
    return ncclSystemError; \
  } \
  *supported = 1; \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO(call, success_retval, name) \
  int ret = call; \
  if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK(call, error_retval, name) \
  int ret = call; \
  if (ret == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PASSTHRU(call) \
  call; \
  return ncclSuccess;

NCCL_PARAM(IbMQpRetryAll, "IB_MQP_RETRY_ALL", 0);
NCCL_PARAM(IbMQpRetryCnt, "IB_MQP_RETRY_CNT", 34);
NCCL_PARAM(IbMQpRetryTimeout, "IB_MQP_RETRY_SLEEP_MSEC", 100); // in milliseconds

#define IBV_ERR_EQ(e, code)        (e == code || e == (-code))
#define IBV_MQP_RETRY_ERRNO(e)     (IBV_ERR_EQ(e, ETIMEDOUT))
#define IBV_MQP_RETRY_ERRNO_ALL(e) (ncclParamIbMQpRetryAll() ? (e != 0) : IBV_MQP_RETRY_ERRNO(e))

ncclResult_t wrap_ibv_fork_init() {
  IBV_INT_CHECK(ibv_fork_init(), -1, "ibv_fork_init");
}

ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices) {
  *ret = ibv_get_device_list(num_devices);
  if (*ret == NULL) *num_devices = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list) {
  IBV_PASSTHRU(ibv_free_device_list(list));
}

const char *wrap_ibv_get_device_name(struct ibv_device *device) {
  return ibv_get_device_name(device);
}

ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device) {
  struct ibv_context* verbsContext = ibv_open_device(device);
  if (verbsContext == NULL) {
    WARN("Call to ibv_open_device failed");
    return ncclSystemError;
  }
  *ret = verbsContext;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_close_device(struct ibv_context *context) {
  IBV_INT_CHECK(ibv_close_device(context), -1, "ibv_close_device");
}

ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) { /*returns 0 on success, and -1 on error*/
  IBV_INT_CHECK(ibv_get_async_event(context, event), -1, "ibv_get_async_event");
}

ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  IBV_PASSTHRU(ibv_ack_async_event(event));
}

ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_query_device(context, device_attr), 0, "ibv_query_device");
}

ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_query_port(context, port_num, port_attr), 0, "ibv_query_port");
}

ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  IBV_INT_CHECK_RET_ERRNO(ibv_query_gid(context, port_num, index, gid), 0, "ibv_query_gid");
}

ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
  IBV_INT_CHECK_RET_ERRNO(ibv_query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

ncclResult_t wrap_mrc_query_qp(struct mrc_qp *mrcQp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr, struct mrc_qp_attr *mrc_attr, int mrc_attr_mask, struct mrc_qp_init_attr *mrc_init_attr) {
  int mrcErrno = mrc_query_qp(mrcQp, attr, attr_mask, mrc_attr, mrc_attr_mask, mrc_init_attr);
  if (mrcErrno != 0) {
    WARN("mrc_query_qp failed with errno %d", mrcErrno);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context) {
  IBV_PTR_CHECK(ibv_alloc_pd(context), *ret, NULL, "ibv_alloc_pd");
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access) {
  IBV_PTR_CHECK(ibv_reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  return ibv_reg_mr(pd, addr, length, access);
}

ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
  if (ret == NULL) return ncclSuccess;
  IBV_PTR_CHECK_ERRNO(ibv_reg_mr_iova2(pd, addr, length, iova, access), *ret, NULL, "ibv_reg_mr_iova2");
}

/* DMA-BUF support */
ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
#if HAVE_DECL_IBV_REG_DMABUF_MR
  IBV_PTR_CHECK_ERRNO(ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access), *ret, NULL, "ibv_reg_dmabuf_mr");
#else
  return ncclSystemError;
#endif
}

struct ibv_mr * wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
#if HAVE_DECL_IBV_REG_DMABUF_MR
  return ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access);
#else
  errno = EOPNOTSUPP; // ncclIbDmaBufSupport() requires this errno being set
  return NULL;
#endif
}

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_dereg_mr(mr), 0, "ibv_dereg_mr");
}

ncclResult_t wrap_mrc_create_cq(struct mrc_cq **ret, struct mrc_context *mrcCtx, int cqe, void *cq_context, struct mrc_comp_channel *channel, int comp_vector) {
  struct mrc_cq* mrcCq = mrc_create_cq(mrcCtx, cqe, cq_context, channel, comp_vector);
  if (mrcCq == NULL) {
    WARN("mrc_create_cq failed");
    return ncclSystemError;
  }
  *ret = mrcCq;
  return ncclSuccess;
}

ncclResult_t wrap_mrc_destroy_cq(struct mrc_cq *cq) {
  int mrcErrno = mrc_destroy_cq(cq);
  if (mrcErrno != 0) {
    WARN("mrc_destroy_cq failed with errno %d", mrcErrno);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_mrc_destroy_qp(struct mrc_qp *mrcQp) {
  // Destroy the QP first: the hint may still be referenced by the QP.
  // Destroying the hint while the QP still holds a reference returns EBUSY.
  int mrcErrno = mrc_destroy_qp(mrcQp);
  if (mrcErrno != 0) {
    WARN("mrc_destroy_qp failed with errno %d", mrcErrno);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_mrc_create_qp(struct mrc_qp **ret, struct mrc_context* mrcCtx, struct mrc_qp_init_attr *mrcQpInitAttr) {
  struct mrc_qp* mrcQp = mrc_create_qp(mrcCtx, mrcQpInitAttr);
  if (mrcQp == NULL) {
    WARN("mrc_create_qp failed");
    return ncclSystemError;
  }
  *ret = mrcQp;

  uint32_t qpn = 0;
  int mrcErrno = mrc_get_qpn(mrcQp, &qpn);
  if (mrcErrno != 0) {
    (void)mrc_destroy_qp(mrcQp);
    WARN("mrc_get_qpn failed with errno %d", mrcErrno);
    return ncclSystemError;
  }
  TRACE(NCCL_NET, "NET/IB : MRC QP created, qpn%u", qpn);

  return ncclSuccess;
}

#define QP_ATTR(attr, userAttr, userFlag, mask) ((userFlag & mask) ? (userAttr) : (attr))

ncclResult_t wrap_mrc_modify_qp(struct mrc_qp* mrcQp, struct ibv_qp_attr* attr, int attr_mask, struct mrc_qp_attr* mrcAttr, int mrcAttrMask) {
  int ret;
  ret = mrc_modify_qp(mrcQp, attr, attr_mask, mrcAttr, mrcAttrMask);
  if (ret) {

    WARN("Call to mrc_modify_qp failed with %d %s", ret, strerror(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_mrc_post_send(struct mrc_qp *mrcQp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
  int ret;
  ret = mrc_post_send(mrcQp, wr, bad_wr);
  if (ret) {
    WARN("Call to mrc_post_send failed with error %s errno %d", strerror(ret), ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_mrc_post_recv(struct mrc_qp *mrcQp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr) {
  int ret;
  ret = mrc_post_recv(mrcQp, wr, bad_wr);
  if (ret) {
    WARN("Call to mrc_post_recv failed with error %s errno %d", strerror(ret), ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_mrc_destroy_qp_hint(struct mrc_qp_hint* qpHint) {
  if (qpHint == NULL) return ncclSuccess;
  int ret = mrc_destroy_qp_hint(qpHint);
  if (ret != 0) {
    WARN("NET/IB : mrc_destroy_qp_hint failed with error %d", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event) {
  *ret = (char *) ibv_event_type_str(event);
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  IBV_PTR_CHECK_ERRNO(ibv_create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq) {
  IBV_INT_CHECK_RET_ERRNO(ibv_destroy_cq(cq), 0, "ibv_destroy_cq");
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  IBV_PTR_CHECK_ERRNO(ibv_create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp");
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp) {
  IBV_INT_CHECK_RET_ERRNO(ibv_destroy_qp(qp), 0, "ibv_destroy_qp");
}

static void ibvQpStateName(enum ibv_qp_state state, char* msg, const size_t len) {
  switch (state) {
  case (IBV_QPS_RESET): snprintf(msg, len, "RESET"); break;
  case (IBV_QPS_INIT): snprintf(msg, len, "INIT"); break;
  case (IBV_QPS_RTR): snprintf(msg, len, "RTR"); break;
  case (IBV_QPS_RTS): snprintf(msg, len, "RTS"); break;
  case (IBV_QPS_SQD): snprintf(msg, len, "SQD"); break;
  case (IBV_QPS_SQE): snprintf(msg, len, "SQE"); break;
  case (IBV_QPS_ERR): snprintf(msg, len, "ERR"); break;
  case (IBV_QPS_UNKNOWN): snprintf(msg, len, "UNKNOWN"); break;
  default: snprintf(msg, len, "NOT RECOGNIZED (%d)", state); break;
  }
}

static void ibvModifyQpLog(struct ibv_qp* qp, enum ibv_qp_state qpState, struct ibv_qp_attr* userAttr, int userFlag, char* msg, size_t msgLen) {
  ncclResult_t res;
  int portNum = -1, gidIndex = -1;
  char localGidName[INET6_ADDRSTRLEN], remoteGidName[INET6_ADDRSTRLEN];
  const char *localGidRes = NULL, *remoteGidRes = NULL;

  char nextState[32], currState[32];
  ibvQpStateName(qp->state, currState, sizeof(currState));
  ibvQpStateName(qpState, nextState, sizeof(nextState));
  char devName[IBV_SYSFS_NAME_MAX] = "";
  snprintf(devName, sizeof(devName), "%s", (qp->pd->context) ? wrap_ibv_get_device_name(qp->pd->context->device) : "N/A");

  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  int attr_mask = IBV_QP_PORT | IBV_QP_AV;
  res = wrap_ibv_query_qp(qp, &attr, attr_mask, &init_attr);
  struct ibv_qp_attr *qpAttr = (res == ncclSuccess) ? &attr : NULL;

  // port info, portAttr can be NULL if not given by the user and query_qp failed
  struct ibv_qp_attr *portAttr = QP_ATTR(qpAttr, userAttr, userFlag, IBV_QP_PORT);
  portNum = portAttr ? portAttr->port_num : -1;

  // address info, avAttr can be NULL if not given by the user and query_qp failed
  struct ibv_qp_attr *avAttr = QP_ATTR(qpAttr, userAttr, userFlag, IBV_QP_AV);
  if (avAttr && avAttr->ah_attr.is_global) {
    union ibv_gid *remoteGid = &avAttr->ah_attr.grh.dgid;
    remoteGidRes = ibvGetGidStr(remoteGid, remoteGidName, sizeof(remoteGidName));
    // we need pd->context to retrieve local GID, skip if not there
    if (!qp->pd->context) goto print;
    gidIndex =  avAttr->ah_attr.grh.sgid_index;
    union ibv_gid localGid;
    NCCLCHECKGOTO(wrap_ibv_query_gid(qp->pd->context, portNum, gidIndex, &localGid), res, print);
    localGidRes = ibvGetGidStr(&localGid, localGidName, sizeof(localGidName));
  }
print:
  snprintf(msg, msgLen, "on dev %s:%d, curr state %s, next state %s, local GID index %d, local GID %s, remote GID %s",
           devName, portNum, currState, nextState, gidIndex, localGidRes ? localGidName : "N/A", remoteGidRes ? remoteGidName : "N/A");
  return;
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  char qpMsg[1024];
  int ret = 0, attempts = 0;
  int maxCnt = (int)ncclParamIbMQpRetryCnt() + 1; // number of attempts = number of retry + 1
  int timeOut = (int)ncclParamIbMQpRetryTimeout();
  do {
    if (attempts > 0) {
      unsigned int sleepTime = timeOut * attempts;
      ibvModifyQpLog(qp, attr->qp_state, attr, attr_mask, qpMsg, sizeof(qpMsg));
      INFO(NCCL_NET, "Call to ibv_modify_qp failed with %d %s, %s, retrying %d/%d after %u msec of sleep", ret, strerror(ret), qpMsg, attempts, maxCnt, sleepTime);
      // sleep before retrying
      struct timespec tv = {.tv_sec = sleepTime / 1000, .tv_nsec = (sleepTime % 1000) * ((long)1e6)};
      nanosleep(&tv, NULL);
    }
    ret = ibv_modify_qp(qp, attr, attr_mask);
    attempts++;
  } while (IBV_MQP_RETRY_ERRNO_ALL(ret) && attempts < maxCnt);
  if (ret != 0) {
    ibvModifyQpLog(qp, attr->qp_state, attr, attr_mask, qpMsg, sizeof(qpMsg));
    WARN("Call to ibv_modify_qp failed with %d %s, %s", ret, strerror(ret), qpMsg);
    return ncclSystemError;
  }
   return ncclSuccess;
 }

ncclResult_t wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
  IBV_INT_CHECK_RET_ERRNO(ibv_post_send(qp, wr, bad_wr), 0, "ibv_post_send");
}

ncclResult_t wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr) {
  IBV_INT_CHECK_RET_ERRNO(ibv_post_recv(qp, wr, bad_wr), 0, "ibv_post_recv");
}

bool wrap_mlx5dv_is_supported(struct ibv_device *device) {
	return mlx5dv_is_supported(device);
}

ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context *context, char *buf, size_t buf_len) {
#if HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH
   int ret;
  ret = mlx5dv_get_data_direct_sysfs_path(context, buf, buf_len);
  if (ret == 0) return ncclSuccess;
  /* ENODEV can happen if the devices is not data-direct but mlx5 is used. It's not an error*/
  if (ret == ENODEV) return ncclInvalidArgument;
  INFO(NCCL_NET, "Call to mlx5dv_get_data_direct_sysfs_path failed with error %s errno %d", strerror(ret), ret);
  return ncclSystemError;
#else
  INFO(NCCL_NET, "Symbol mlx5dv_get_data_direct_sysfs_path in rdma-core library");
  return ncclSystemError;
#endif
}

  /* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access) {
#if HAVE_DECL_MLX5DV_REG_DMABUF_MR
  *ret = mlx5dv_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
  if (*ret == NULL) {
    WARN("Call to mlx5dv_reg_dmabuf_mr failed with error %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
#else
  return ncclSystemError;
#endif
}

struct ibv_mr * wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int is_data_direct) {
#if HAVE_DECL_MLX5DV_REG_DMABUF_MR
  int mlx5_access = (is_data_direct) ? MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT : 0;
  return mlx5dv_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
#else
  errno = EOPNOTSUPP;
  return NULL;
#endif
}

ncclResult_t wrap_mrc_create_context(struct ibv_context* context, struct mrc_context** mrcContext) {
  if (context == NULL) {
    WARN("NET/IB : wrap_mrc_create_context called with NULL ibv_context");
    return ncclInternalError;
  }

  struct mrc_attr attr;
  int mrcSupported = 0;
  int mrcErrno = mrc_query_device(context, &attr, &mrcSupported);
  if (mrcErrno != 0) {
    WARN("NET/IB : mrc_query_device returned %d", mrcErrno);
    return ncclSystemError;
  }
  if (!mrcSupported) {
    WARN("NET/IB : MRC not supported on this device, MRC plugin cannot be used");
    return ncclSystemError;
  }

  struct mrc_context_attr mrcContextAttr;
  memset(&mrcContextAttr, 0, sizeof(mrcContextAttr));
#ifdef MRC_API_VER_USED
  mrcContextAttr.mrc_api_version_used = MRC_API_VER_USED;
#endif
  *mrcContext = mrc_create_context(context, &mrcContextAttr);
  if (*mrcContext == NULL) {
    WARN("NET/IB : MRC context creation failed");
    return ncclSystemError;
  }

  INFO(NCCL_NET, "NET/IB : MRC context created");
  return ncclSuccess;
}
