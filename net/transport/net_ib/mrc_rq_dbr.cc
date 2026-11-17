// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#define DOCA_ALLOW_EXPERIMENTAL_API

#include "mrc_rq_dbr.h"

#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <atomic>

#include <doca_verbs.h>
#include <mrc.h>

#include "debug.h"
#include "param.h"

NCCL_PARAM(MrcRqDirectDbr, "MRC_RQ_DIRECT_DBR", 1);

#define NCCL_MLX5_RCV_DBR 0
#define NCCL_MLX5_INVALID_LKEY 0x100

// Order WQE writes before the doorbell-record store that makes them visible to the device
// (rdma-core's udma_to_device_barrier()).
#if defined(__aarch64__)
#define ncclIbMrcToDeviceBarrier() asm volatile("dmb oshst" ::: "memory")
#elif defined(__x86_64__)
#define ncclIbMrcToDeviceBarrier() asm volatile("" ::: "memory")
#else
#define ncclIbMrcToDeviceBarrier() __atomic_thread_fence(__ATOMIC_RELEASE)
#endif

bool ncclIbMrcRqDbrEnabled() { return ncclParamMrcRqDirectDbr() != 0; }

namespace {
constexpr int kMaxCapturedQps = 4;
// Per thread, so concurrent connection setups on different proxy threads do not race.
struct QpCapture {
  bool armed;
  int n;
  struct doca_verbs_qp* qps[kMaxCapturedQps];
};
thread_local QpCapture qpCapture;

typedef doca_error_t (*QpCreateFn)(struct doca_verbs_context*, struct doca_verbs_qp_init_attr*,
                                   struct doca_verbs_qp**);
std::atomic<QpCreateFn> realQpCreate{nullptr};
} // namespace

// Pass-through interposer: libnv_mrc resolves doca_verbs_qp_create through the plugin's
// lookup scope, so exporting it here lets mrc_create_qp() hand us the DOCA Verbs QP it
// creates. QP attributes are never modified.
extern "C" __attribute__((visibility("default"))) doca_error_t doca_verbs_qp_create(
  struct doca_verbs_context* verbsContext, struct doca_verbs_qp_init_attr* initAttr, struct doca_verbs_qp** verbsQp) {
  QpCreateFn fn = realQpCreate.load(std::memory_order_acquire);
  if (fn == nullptr) {
    fn = (QpCreateFn)dlvsym(RTLD_NEXT, "doca_verbs_qp_create", "EXPERIMENTAL");
    if (fn == nullptr) fn = (QpCreateFn)dlsym(RTLD_NEXT, "doca_verbs_qp_create");
    if (fn == nullptr) {
      WARN("NET/MRC: cannot resolve the DOCA Verbs doca_verbs_qp_create: %s", dlerror());
      return DOCA_ERROR_NOT_FOUND;
    }
    realQpCreate.store(fn, std::memory_order_release);
  }
  doca_error_t ret = fn(verbsContext, initAttr, verbsQp);
  if (qpCapture.armed && ret == DOCA_SUCCESS && qpCapture.n < kMaxCapturedQps) {
    qpCapture.qps[qpCapture.n++] = *verbsQp;
  }
  return ret;
}

static void ncclIbMrcRqDbrSetup(struct ncclIbMrcRqDbr* rqDbr, uint32_t qpn, uint32_t maxRecvWr) {
  struct doca_verbs_qp* vqp = nullptr;
  for (int i = 0; i < qpCapture.n; i++) {
    if (doca_verbs_qp_get_qpn(qpCapture.qps[i]) == qpn) {
      vqp = qpCapture.qps[i];
      break;
    }
  }
  if (vqp == nullptr) {
    INFO(NCCL_NET, "NET/MRC: qp_num=%u: DOCA Verbs QP not captured (%d candidates), using mrc_post_recv", qpn,
         qpCapture.n);
    return;
  }

  void *sqBuf = nullptr, *rqBuf = nullptr;
  uint32_t sqEntries = 0, rqEntries = 0, wqeSize = 0;
  doca_verbs_qp_get_wq(vqp, &sqBuf, &sqEntries, &rqBuf, &rqEntries, &wqeSize);
  volatile uint32_t* dbrRec = (volatile uint32_t*)doca_verbs_qp_get_dbr_addr(vqp);
  // The producer index wraps at 16 bits and indexes the ring modulo its size.
  bool pow2 = rqEntries != 0 && (rqEntries & (rqEntries - 1)) == 0;
  if (dbrRec == nullptr || rqBuf == nullptr || !pow2 || rqEntries < maxRecvWr || rqEntries > 0x8000 ||
      wqeSize < 16) {
    INFO(NCCL_NET,
         "NET/MRC: qp_num=%u: unexpected RQ geometry (dbr=%p buf=%p entries=%u wqe_size=%u max_recv_wr=%u), using "
         "mrc_post_recv",
         qpn, (void*)dbrRec, rqBuf, rqEntries, wqeSize, maxRecvWr);
    return;
  }
  rqDbr->dbrRec = &dbrRec[NCCL_MLX5_RCV_DBR];
  rqDbr->rqBuf = rqBuf;
  rqDbr->rqEntries = rqEntries;
  rqDbr->wqeSize = wqeSize;
}

ncclResult_t ncclIbMrcCreateQpCaptureRq(struct mrc_qp** qp, struct mrc_context* ctx, struct mrc_qp_init_attr* attr,
                                        bool wantRqDbr, struct ncclIbMrcRqDbr* rqDbr) {
  memset(rqDbr, 0, sizeof(*rqDbr));
  qpCapture.n = 0;
  qpCapture.armed = wantRqDbr;
  *qp = mrc_create_qp(ctx, attr);
  qpCapture.armed = false;
  if (*qp == nullptr) {
    WARN("NET/IB : mrc_create_qp failed with error %s", strerror(errno));
    return ncclSystemError;
  }
  if (wantRqDbr) {
    uint32_t qpn = 0;
    if (mrc_get_qpn(*qp, &qpn) == 0) {
      ncclIbMrcRqDbrSetup(rqDbr, qpn, attr->cap.max_recv_wr);
    }
  }
  qpCapture.n = 0;
  return ncclSuccess;
}

ncclResult_t ncclIbMrcRqDbrArm(struct ncclIbMrcRqDbr* rqDbr, uint32_t qpn, uint32_t count, bool* armed) {
  *armed = false;
  if (rqDbr->rqBuf == nullptr) return ncclSuccess;
  if (count > rqDbr->rqEntries) {
    WARN("NET/MRC: qp_num=%u: cannot arm %u receive WQEs on an RQ of %u entries", qpn, count, rqDbr->rqEntries);
    return ncclInternalError;
  }
  if (rqDbr->dbr == nullptr) {
    // RDMA WRITE-with-IMM scatters nothing on the responder, so every RQ WQE is the same
    // null data segment that mrc_post_recv() writes for a zero-SGE WR. The device never
    // modifies WQEs, so they are written once and only the producer index moves after this.
    for (uint32_t i = 0; i < rqDbr->rqEntries; i++) {
      volatile uint32_t* seg = (volatile uint32_t*)((char*)rqDbr->rqBuf + (size_t)i * rqDbr->wqeSize);
      seg[0] = htobe32(0);                      // byte_count
      seg[1] = htobe32(NCCL_MLX5_INVALID_LKEY); // lkey: end of scatter list
      seg[2] = 0;                               // addr
      seg[3] = 0;
    }
    rqDbr->pi = be32toh(*rqDbr->dbrRec) & 0xffff;
    rqDbr->dbr = rqDbr->dbrRec;
    INFO(NCCL_NET, "NET/MRC: qp_num=%u: RQ armed via direct doorbell record (entries=%u wqe_size=%u pi=%u)", qpn,
         rqDbr->rqEntries, rqDbr->wqeSize, rqDbr->pi);
  }
  ncclIbMrcToDeviceBarrier();
  rqDbr->pi += count;
  *rqDbr->dbr = htobe32(rqDbr->pi & 0xffff);
  *armed = true;
  return ncclSuccess;
}
