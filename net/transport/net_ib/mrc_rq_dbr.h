// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Direct receive-queue doorbell-record (DBR) re-arm for MRC data QPs.
//
// Receivers of RDMA WRITE-with-IMM need no receive buffer, so every RQ WQE is
// the same null scatter entry and completions are matched by imm_data, not by
// wr_id. Instead of calling mrc_post_recv() for every consumed WQE, the RQ
// WQEs are written once and the RQ is re-armed by publishing the new producer
// index to the QP's doorbell record (the UCX uct_rc_mlx5_iface_update_srq_res()
// pattern, prototyped in mrc_wimm_proto).
//
// The DOCA Verbs QP that libnv_mrc creates is not exposed by mrc.h; it is
// captured by a pass-through doca_verbs_qp_create() exported by this plugin.
// If the capture or the RQ geometry check fails, the QP keeps using
// mrc_post_recv(). The two modes are never mixed on one QP.

#ifndef NET_IB_MRC_RQ_DBR_H_
#define NET_IB_MRC_RQ_DBR_H_

#include <endian.h>
#include <stdint.h>
#include "nccl.h"

struct ncclIbQp;
struct mrc_qp;
struct mrc_context;
struct mrc_qp_init_attr;

struct ncclIbMrcRqDbr {
  volatile uint32_t* dbr;    // Non-NULL once the RQ is armed through the doorbell record.
  volatile uint32_t* dbrRec; // RQ doorbell record captured at QP creation.
  void* rqBuf;               // NULL if direct re-arm is unavailable for this QP.
  uint32_t rqEntries;
  uint32_t wqeSize;
  uint32_t pi;               // Software producer index (only the low 16 bits are published).
};

// Whether NCCL_MRC_RQ_DIRECT_DBR is enabled.
bool ncclIbMrcRqDbrEnabled();

// Create an MRC QP and, when wantRqDbr is set, capture its RQ buffer and
// doorbell record into rqDbr (left zeroed when unavailable).
ncclResult_t ncclIbMrcCreateQpCaptureRq(struct mrc_qp** qp, struct mrc_context* ctx,
                                        struct mrc_qp_init_attr* attr, bool wantRqDbr,
                                        struct ncclIbMrcRqDbr* rqDbr);

// Arm count RQ WQEs on a QP. Uses the direct DBR when available (initializing
// the WQEs on first use), otherwise reports *armed = false and the caller must
// post with mrc_post_recv().
ncclResult_t ncclIbMrcRqDbrArm(struct ncclIbMrcRqDbr* rqDbr, uint32_t qpn, uint32_t count, bool* armed);

// Hot path: re-arm one consumed WQE. Only valid once ncclIbMrcRqDbrArm() has
// armed the QP. The consumed WQE is unchanged and the CQE that freed it has
// already been read, so no fence is needed before the single 4-byte store.
static inline void ncclIbMrcRqDbrRepost(struct ncclIbMrcRqDbr* rqDbr) {
  rqDbr->pi++;
  *rqDbr->dbr = htobe32(rqDbr->pi & 0xffff);
}

#endif
