/*************************************************************************
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2015-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NCCL_IBVWRAP_H_
#define NCCL_IBVWRAP_H_
#include <mrc.h>
#include "config.h"
#include "core.h"
#include "utils.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#if HAVE_INFINIBAND_MLX5DV_H
#include <infiniband/mlx5dv.h>
#endif


ncclResult_t wrap_ibv_fork_init(void);
ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices);
ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list);
const char *wrap_ibv_get_device_name(struct ibv_device *device);
ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device);
ncclResult_t wrap_ibv_close_device(struct ibv_context *context);
ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event);
ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event);
ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr);
ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr);
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context);
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd);
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access);
struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access);
/* DMA-BUF support */
ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access);
struct ibv_mr * wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access);
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr);
ncclResult_t wrap_ibv_create_comp_channel(struct ibv_comp_channel **ret, struct ibv_context *context);
ncclResult_t wrap_ibv_destroy_comp_channel(struct ibv_comp_channel *channel);

ncclResult_t wrap_mrc_create_context(struct ibv_context* context, struct mrc_context** mrcContext);
ncclResult_t wrap_mrc_create_cq(struct mrc_cq **ret, struct mrc_context *mrcCtx, int cqe, void *cq_context, struct mrc_comp_channel *channel, int comp_vector);
ncclResult_t wrap_mrc_destroy_cq(struct mrc_cq *cq);
static inline ncclResult_t wrap_mrc_poll_cq(struct mrc_cq *cq, int num_entries, struct ibv_wc *wc, int* num_done) {
  int done = mrc_poll_cq(cq, num_entries, wc);
  if (done < 0) {
    WARN("Call to mrc_poll_cq() returned %d", done);
    return ncclSystemError;
  }
  *num_done = done;
  return ncclSuccess;
}
ncclResult_t wrap_mrc_create_qp(struct mrc_qp **ret, struct mrc_context* mrcCtx, struct mrc_qp_init_attr *mrcQpInitAttr);
ncclResult_t wrap_mrc_modify_qp(struct mrc_qp* mrcQp, struct ibv_qp_attr* attr, int attr_mask, struct mrc_qp_attr* mrcAttr, int mrcAttrMask);
ncclResult_t wrap_mrc_destroy_qp(struct mrc_qp *mrcQp);
ncclResult_t wrap_mrc_query_qp(struct mrc_qp *mrcQp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr, struct mrc_qp_attr *mrc_attr, int mrc_attr_mask, struct mrc_qp_init_attr *mrc_init_attr);
ncclResult_t wrap_mrc_post_send(struct mrc_qp *mrcQp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
ncclResult_t wrap_mrc_post_recv(struct mrc_qp *mrcQp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);
ncclResult_t wrap_mrc_destroy_qp_hint(struct mrc_qp_hint* qpHint);

ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event);

ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq);
static inline ncclResult_t wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc, int* num_done) {
  int done = ibv_poll_cq(cq, num_entries, wc);
  if (done < 0) {
    WARN("Call to ibv_poll_cq() returned %d", done);
    return ncclSystemError;
  }
  *num_done = done;
  return ncclSuccess;
}
ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp);
ncclResult_t wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
ncclResult_t wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);


// converts a GID into a readable string. On success, returns a non-null pointer to gidStr.
// NULL is returned if there was an error, with errno set to indicate the error.
// errno = ENOSPC if the converted string would exceed strLen.
static inline const char* ibvGetGidStr(union ibv_gid* gid, char* gidStr, size_t strLen) {
  // GID is a 16B handle, to convert it to a readable form, we use inet_ntop
  // sizeof(ibv_gid) == sizeof(struct in6_addr), so using AF_INET6
  NCCL_STATIC_ASSERT(sizeof(union ibv_gid) == sizeof(struct in6_addr), "the sizeof struct ibv_gid must be the size of struct in6_addr");
  return inet_ntop(AF_INET6, gid->raw, gidStr, strLen);
}

bool wrap_mlx5dv_is_supported(struct ibv_device *device);
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context *context, char *buf, size_t buf_len);
/* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);
struct ibv_mr * wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);

#endif //End include guard
