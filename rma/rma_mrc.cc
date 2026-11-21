// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NCCL RMA plugin (ncclRmaPlugin_v17) over MRC. NCCL's GIN CPU proxy (NCCL_GIN_TYPE=2) and its host RMA API
// run on top of it. Derived from the IB RMA proxy of NCCL v2.32.6-1 (src/transport/net_ib/gin.cc), using the
// MRC QPs and CQs of this plugin's net_ib.
//
// MRC has no RDMA atomics or reads, so a signal travels as the immediate of an RDMA Write-with-Immediate,
// which also carries the put payload. The target's progress (rmaProgress) reaps the receive completion, which
// MRC generates only once the whole message is placed, and applies the signal through a CPU mapping of the
// signal buffer. A get is a receiver-driven write: the requester sends a descriptor, and the data owner's
// progress writes the data back (see mrcRmaIGet).

#include <atomic>
#include <limits.h>
#include <mutex>
#include <new>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <time.h>
#include <unordered_map>

#include "transport/net_ib/common.h"
#include "transport/net_ib/p2p_resiliency.h"
#include "gdrwrap.h"
#include "param.h"
#include "utils.h"
#include "rma/rma_v17.h"

// The plugin owns its GDRCopy handle; libnccl's is not visible to plugins.
gdr_t ncclGdrCopy = NULL;

NCCL_PARAM(GinIbTc, "GIN_IB_TC", -1);
extern int64_t ncclParamIbTc();
extern int64_t ncclParamIbSl();
NCCL_PARAM(RmaIbWrBatchsize, "RMA_IB_WR_BATCHSIZE", 64);
NCCL_PARAM(MrcRmaPollCount, "MRC_RMA_POLL_COUNT", 16);
NCCL_PARAM(MrcRmaSignalMapMax, "MRC_RMA_SIGNAL_MAP_MAX", 64 << 20);
NCCL_PARAM(MrcRmaGetSlots, "MRC_RMA_GET_SLOTS", 32);

static const int MRC_RMA_ALLGATHER_TAG = 0xa0;

/* ── Signal immediate layout ──────────────────────────────────────────────
 *
 *   31      26 25                          M+2  M+1  M  M-1         0
 *  [ MR id (6) | slot index (S bits)          | exp (2) | mantissa (M) ]
 *
 * The MR id selects one of the signal buffers registered on the collective comm, S is sized from that
 * buffer's slot count, and M = 24 - S. The value added is mantissa << (exp * M); a value that needs more
 * than one base-2^M digit is sent as several immediates, at most MRC_RMA_SIG_MAX_DIGITS.
 */
#define MRC_RMA_SIG_ID_BITS 6
#define MRC_RMA_SIG_IDS (1 << MRC_RMA_SIG_ID_BITS)
#define MRC_RMA_SIG_EXP_BITS 2
#define MRC_RMA_SIG_MAX_DIGITS (1 << MRC_RMA_SIG_EXP_BITS)
#define MRC_RMA_SIG_ID_SHIFT (32 - MRC_RMA_SIG_ID_BITS)
#define MRC_RMA_SIG_FIELD_BITS (MRC_RMA_SIG_ID_SHIFT - MRC_RMA_SIG_EXP_BITS)
#define MRC_RMA_SIG_MAX_SLOT_BITS (MRC_RMA_SIG_FIELD_BITS - 1)

/* ── Control immediate layout (get) ───────────────────────────────────────
 *
 *   31      26 25  24 23                            0
 *  [ 63 (6)   | op (2) | descriptor slot (24)        ]
 *
 * The top MR id is reserved for control messages, so MRC_RMA_SIG_BUFS ids remain for signal buffers.
 * GET_REQ asks the target to serve the descriptor in its inbox slot; GET_RESP carries the data back and
 * completes the requester's get in that slot.
 */
#define MRC_RMA_CTRL_ID (MRC_RMA_SIG_IDS - 1)
#define MRC_RMA_SIG_BUFS MRC_RMA_CTRL_ID
#define MRC_RMA_CTRL_OP_SHIFT (MRC_RMA_SIG_ID_SHIFT - 2)
#define MRC_RMA_CTRL_ARG_MASK ((1u << MRC_RMA_CTRL_OP_SHIFT) - 1)
#define MRC_RMA_CTRL_GET_REQ 0u
#define MRC_RMA_CTRL_GET_RESP 1u
#define MRC_RMA_CTRL_IMM(op, arg) \
  (((uint32_t)MRC_RMA_CTRL_ID << MRC_RMA_SIG_ID_SHIFT) | ((uint32_t)(op) << MRC_RMA_CTRL_OP_SHIFT) | (uint32_t)(arg))
// Get responses hold no ncclIbRequest (NCCL may use all of a comm's requests); their completions carry this wr_id.
#define MRC_RMA_WR_ID_GET_RESP (1ull << 63)
#define MRC_RMA_GET_SLOT_FREE UINT16_MAX
#define MRC_RMA_GET_MAX_BYTES (1ull << 31)
// Get responses in flight per peer and context. They share the send QP and CQ with up to NET_IB_MAX_REQUESTS of
// NCCL's requests, which those are sized for, so further requests wait in the inbox until responses complete.
#define MRC_RMA_GET_RESP_MAX 16

// A get request as the requester writes it into the target's inbox. Memory regions are named by their
// collective registration id, the same on every rank of the collective comm.
struct mrcRmaGetDesc {
  uint32_t srcMrId;  // target's memory, read
  uint32_t dstMrId;  // requester's memory, written
  uint64_t srcOff;
  uint64_t dstOff;
  uint64_t size;
};

// A get waiting for a free descriptor slot to `rank`; posted when a response frees one (mrcRmaCompleteGet).
struct mrcRmaQueuedGet {
  uint16_t reqIdx;
  struct mrcRmaGetDesc desc;
};

// One put plus up to MRC_RMA_SIG_MAX_DIGITS signal immediates; the payload rides the last one.
#define MRC_RMA_MAX_WRS_PER_OP MRC_RMA_SIG_MAX_DIGITS
#define MRC_RMA_WR_BATCH_MAX 64

// Receive WRs kept posted per peer QP; each incoming signal consumes one. Refilled in batches of REFILL.
// When net_ib pre-posts its own receive WRs at accept (NCCL_MRC_PREPOST_RECEIVE_WORK_REQUESTS), those fill
// the queue and become the depth instead.
#define MRC_RMA_RECV_DEPTH 128
#define MRC_RMA_RECV_DEPTH_MAX NET_IB_MAX_REQUESTS
#define MRC_RMA_RECV_REFILL 32
#define MRC_RMA_POLL_MAX 64

struct mrcRmaMrHandle {
  struct ncclIbMrHandle* mrHandle;
  uintptr_t* baseVas;
  uint32_t* rkeys;
  size_t bytes;
  uint32_t mrId;  // collective registration id, see mrcRmaCollComm::mrById
  // Signal target state, set when the MR got a signal id (-1 otherwise).
  int sigId;
  bool sigIdsExhausted;  // eligible as a signal target, but every id was in use at registration
  int slotBits;
  bool neverReset;
  volatile uint64_t* sigCpu;  // CPU view of the local buffer; NULL if it could not be mapped
  uint64_t* shadow;           // current values of a never-reset buffer, owned by the progress thread
  void* gdrMap;
  size_t gdrMapSize;
  gdr_mh_t gdrMh;
};

struct mrcRmaCollComm {
  void* ctx;
  int rank;
  int nranks;
  int dev;
  void* recvComm;
  void* sendComm;
  volatile uint32_t* abortFlag;
  bool dataDirect;       // the device writes GPU memory through Data Direct
  uint32_t sigPinFlags;  // GDRCopy pin flags for signal buffers, see mrcRmaSetupSignalPath
  std::mutex sigMutex;  // guards sigMrs updates and mrById
  std::atomic<struct mrcRmaMrHandle*> sigMrs[MRC_RMA_SIG_BUFS];
  // Every symmetric registration gets the next id; registration is collective, so ids agree across ranks.
  uint32_t nextMrId;
  std::unordered_map<uint32_t, struct mrcRmaMrHandle*> mrById;
};

struct mrcRmaWrBatch {
  struct mrc_qp* qp;
  int nWrs;
  struct ibv_send_wr wrs[MRC_RMA_WR_BATCH_MAX];
  struct ibv_sge sges[MRC_RMA_WR_BATCH_MAX];
};

struct mrcRmaCtx {
  struct mrcRmaCollComm* cComm;
  void** fullRecvComm;
  void** fullSendComm;
  uint32_t* recvOutstanding;
  uint32_t* recvDepth;
  int rank, nranks;
  int nContexts;
  int wrBatchsize;
  ncclResult_t asyncError;
  struct mrcRmaWrBatch batch;
  // Get state. getBuf holds this context's inbox and outbox, each [nranks][getSlots] descriptors, inside the
  // get MR shared by all contexts of the rmaCtx (owned by context 0).
  int getSlots;                  // descriptor slots per peer; 0 disables get
  char* getBuf;
  uint64_t getBufOff;            // offset of getBuf in getMr
  struct mrcRmaMrHandle* getMr;
  uint16_t* getSlotReq;          // [nranks][getSlots]: index of the request waiting on the slot, or FREE
  uint32_t* getCursor;           // [nranks]: next slot to try
  uint8_t* getDone;              // [nranks][NET_IB_MAX_REQUESTS]: the response to this request arrived
  uint32_t* getRespPending;      // [nranks]: responses posted to the peer whose completion is not reaped yet
  std::deque<struct mrcRmaQueuedGet>* getQueue;  // [nranks]: gets waiting for a slot, in issue order
  std::deque<uint32_t>* getServeQueue;           // [nranks]: inbox slots waiting for response credit
};

static inline size_t mrcRmaGetInboxOff(struct mrcRmaCtx* ctx, int requester, int slot) {
  return ((size_t)requester * ctx->getSlots + slot) * sizeof(struct mrcRmaGetDesc);
}
static inline size_t mrcRmaGetOutboxOff(struct mrcRmaCtx* ctx, int target, int slot) {
  return ((size_t)(ctx->nranks + target) * ctx->getSlots + slot) * sizeof(struct mrcRmaGetDesc);
}

static inline bool mrcRmaAborted(volatile uint32_t* abortFlag) {
  return abortFlag && COMPILER_ATOMIC_LOAD(abortFlag, std::memory_order_acquire) != 0;
}

/* ── Init, devices, properties ─────────────────────────────────────────── */

static std::once_flag mrcRmaGdrOnce;

static ncclResult_t mrcRmaInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  (void)commId;
  ncclMrcSetLogger(logFunction);
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError;
  if (ncclIbGdrSupport() != ncclSuccess && ncclIbDmaBufSupport(0) != ncclSuccess) {
    INFO(NCCL_INIT | NCCL_NET, "NET/MRC/RMA: neither peermem nor DMA-BUF is available, disabling MRC RMA");
    ncclIbFinalizeDevices();
    return ncclInternalError;
  }
  // Signals are applied through GDRCopy mappings of the signal buffers.
  std::call_once(mrcRmaGdrOnce, []() {
    ncclGdrCopy = ncclGdrInit();
    if (ncclGdrCopy == NULL) {
      WARN("NET/MRC/RMA: GDRCopy is unavailable; GPU signal buffers cannot be mapped and signals to them will fail");
    }
  });
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  INFO(NCCL_INIT | NCCL_NET, "NET/MRC/RMA: MRC RMA plugin initialized (%d devices, GDRCopy %s)", ncclNMergedIbDevs,
       ncclGdrCopy ? "on" : "off");
  return ncclSuccess;
}

static ncclResult_t mrcRmaFinalize(void* ctx) {
  free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t mrcRmaGetRmaProperties(void* collComm, ncclRmaProperties_v16_t* rmaProps) {
  (void)collComm;
  // Signals are applied by the target's CPU from receive completions, so a signal does not flush the
  // puts other peers have in flight.
  rmaProps->flushesAllPutsOnAnySignal = false;
  return ncclSuccess;
}

static ncclResult_t mrcRmaGetProperties(int dev, ncclNetProperties_v12_t* props) {
  NCCLCHECK(ncclIbGetProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

/* ── Collective comm: a ring of send/recv comms for setup collectives ──── */

static ncclResult_t mrcRmaAllGather(struct mrcRmaCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer = cComm->rank;
  int step = 0;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &rMhandle), status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf, cComm->nranks * len, NCCL_PTR_HOST, &sMhandle), status, out);

  memcpy((void*)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (step = 0; step < cComm->nranks - 1; step++) {
    int rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    void* rbuf = (void*)((uintptr_t)recvBuf + rpeer * len);
    int tag = MRC_RMA_ALLGATHER_TAG;
    int sendDone = 0, recvDone = 0;
    while (!sendDone || !recvDone) {
      if (!srequest && !sendDone) {
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm, (void*)((uintptr_t)recvBuf + speer * len), len, tag, sMhandle,
                                      NULL, &srequest),
                      status, out);
      }
      if (!rrequest && !recvDone) {
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len, &tag, &rMhandle, NULL, &rrequest), status, out);
      }
      if (rrequest) {
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &recvDone, NULL), status, out);
        if (recvDone) rrequest = NULL;
      }
      if (srequest) {
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &sendDone, NULL), status, out);
        if (sendDone) srequest = NULL;
      }
      if (mrcRmaAborted(cComm->abortFlag)) {
        INFO(NCCL_NET, "NET/MRC/RMA: allgather aborted on rank %d/%d at step %d", cComm->rank, cComm->nranks, step);
        status = ncclInternalError;
        goto out;
      }
    }
    speer = rpeer;
  }

out:
  if (status != ncclSuccess) {
    WARN("NET/MRC/RMA: allgather failed (res=%d) on rank %d/%d at step %d/%d", status, cComm->rank, cComm->nranks,
         step, cComm->nranks - 1);
  }
  if (rMhandle) ncclNetIb.deregMr(cComm->recvComm, rMhandle);
  if (sMhandle) ncclNetIb.deregMr(cComm->sendComm, sMhandle);
  return status;
}

static ncclResult_t mrcRmaBarrier(struct mrcRmaCollComm* cComm) {
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks * sizeof(int)));
  ncclResult_t ret = mrcRmaAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int));
  free(dummy);
  return ret;
}

static int mrcRmaTrafficClass() {
  return ncclParamGinIbTc() != NCCL_PARAM_VAL_AUTO ? ncclParamGinIbTc() : ncclParamIbTc();
}

static ncclResult_t mrcRmaGdrSupport(int dev) {
  if (ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(dev) == ncclSuccess) return ncclSuccess;
  WARN("NET/MRC/RMA: neither peermem nor DMA-BUF is available for device %d", dev);
  return ncclInvalidUsage;
}

static ncclResult_t mrcRmaCloseColl(void* collComm);
static ncclResult_t mrcRmaSetupSignalPath(struct mrcRmaCollComm* cComm);

static ncclResult_t mrcRmaConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                                  volatile uint32_t* abortFlag) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  ncclResult_t ret = ncclSuccess;
  *collComm = NULL;
  NCCLCHECK(mrcRmaGdrSupport(lComm->dev));

  struct mrcRmaCollComm* cComm = new (std::nothrow) mrcRmaCollComm();
  if (cComm == NULL) return ncclSystemError;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;
  cComm->dev = lComm->dev;
  cComm->abortFlag = abortFlag;

  int next = (rank + 1) % nranks;
  NCCLCHECKGOTO(mrcRmaSetupSignalPath(cComm), ret, fail);
  do {
    if (cComm->sendComm == NULL) {
      NCCLCHECKGOTO(ncclIbConnectImpl(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL, /*nQpsPerDev*/ 1,
                                      mrcRmaTrafficClass()),
                    ret, fail);
    }
    if (cComm->recvComm == NULL) {
      NCCLCHECKGOTO(ncclIbAcceptImpl(lComm, &cComm->recvComm, NULL, /*nQpsPerDev*/ 1), ret, fail);
    }
    if (mrcRmaAborted(abortFlag)) {
      INFO(NCCL_NET, "NET/MRC/RMA: connect aborted on rank %d/%d", rank, nranks);
      ret = ncclInternalError;
      goto fail;
    }
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  *collComm = cComm;
  return ncclSuccess;

fail:
  mrcRmaCloseColl(cComm);
  return ret;
}

static ncclResult_t mrcRmaCloseColl(void* collComm) {
  struct mrcRmaCollComm* cComm = (struct mrcRmaCollComm*)collComm;
  if (cComm == NULL) return ncclSuccess;
  ncclResult_t ret = ncclSuccess;
  if (cComm->recvComm) NCCLCHECKIGNORE(ncclNetIb.closeRecv(cComm->recvComm), ret);
  if (cComm->sendComm) NCCLCHECKIGNORE(ncclNetIb.closeSend(cComm->sendComm), ret);
  delete cComm;
  return ret;
}

/* ── Memory registration and signal buffers ───────────────────────────── */

// True when the PCI devices at sysfs paths a and b share the same root port, i.e. sit under one PCIe switch
// hierarchy and do not cross the host bridge.
static bool mrcRmaPciNear(const char* a, const char* b) {
  char *ra = realpath(a, NULL), *rb = realpath(b, NULL);
  bool near = false;
  if (ra && rb) {
    // Paths look like /sys/devices/pciDDDD:BB/<root port>/<switch port>/.../<device>
    int shared = 0;
    const char *pa = ra, *pb = rb;
    while (*pa && *pa == *pb) {
      if (*pa == '/' && pa != ra) shared++;
      pa++, pb++;
    }
    if (*pa == '\0' && (*pb == '/' || *pb == '\0')) shared++;
    // Shared components: "sys", "devices", "pciDDDD:BB", root port.
    near = shared >= 4;
  }
  free(ra);
  free(rb);
  return near;
}

/* The CPU's signal store must reach GPU memory on the same path as the NIC's payload writes, or the GPU can see a
 * signal before its data. This follows NCCL's choice for its own GDRCopy sync word (ncclTopoNeedFlush and
 * ncclGdcPinFlag, used with NCCL_GDRCOPY_SYNC_ENABLE):
 *  - regular NIC: the payload reaches GPU memory over C2C on a C2C-coherent GPU (PCIe otherwise), the same path
 *    as a default GDRCopy mapping;
 *  - Data Direct NIC on a C2C-coherent GPU: the payload arrives over the PCIe switch the NIC shares with the GPU,
 *    so signal mappings are forced onto PCIe (GDR_PIN_FLAG_FORCE_PCIE, GDRCopy >= 2.5). A Data Direct NIC that is
 *    not behind the GPU's root port, or a merged device mixing both kinds, has no ordered path and is refused.
 */
static ncclResult_t mrcRmaSetupSignalPath(struct mrcRmaCollComm* cComm) {
  cComm->dataDirect = false;
  cComm->sigPinFlags = GDR_PIN_FLAG_DEFAULT;
  struct ncclIbMergedDev* mDev = &ncclIbMergedDevs[cComm->dev];
  int nDataDirect = 0;
  for (int i = 0; i < mDev->vProps.ndevs; i++) {
    struct ncclIbDev* ibDev = &ncclIbDevs[mDev->vProps.devs[i]];
    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ibDev->capsProvider.mlx5.dataDirect) nDataDirect++;
  }
  if (nDataDirect == 0) return ncclSuccess;
  if (nDataDirect != mDev->vProps.ndevs) {
    WARN("NET/MRC/RMA: device %s mixes Data Direct and regular NICs; their writes reach GPU memory on different "
         "paths, so signals cannot be ordered behind data",
         mDev->devName);
    return ncclInvalidUsage;
  }
  cComm->dataDirect = true;

  int cudaDev, c2cCoherent = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUDACHECK(cudaDeviceGetAttribute(&c2cCoherent, cudaDevAttrPageableMemoryAccessUsesHostPageTables, cudaDev));
  if (!c2cCoherent) return ncclSuccess;

  int domain = 0, bus = 0, device = 0;
  CUDACHECK(cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, cudaDev));
  CUDACHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, cudaDev));
  CUDACHECK(cudaDeviceGetAttribute(&device, cudaDevAttrPciDeviceId, cudaDev));
  char gpuPath[PATH_MAX];
  snprintf(gpuPath, sizeof(gpuPath), "/sys/bus/pci/devices/%04x:%02x:%02x.0", domain, bus, device);
  for (int i = 0; i < mDev->vProps.ndevs; i++) {
    struct ncclIbDev* ibDev = &ncclIbDevs[mDev->vProps.devs[i]];
    if (ibDev->pciPath == NULL || !mrcRmaPciNear(gpuPath, ibDev->pciPath)) {
      WARN("NET/MRC/RMA: Data Direct NIC %s (%s) is not behind the root port of GPU %d (%s), so its writes are not "
           "ordered with signal stores. Use the GPU's local NIC or set NCCL_IB_DATA_DIRECT=0",
           ibDev->devName, ibDev->pciPath ? ibDev->pciPath : "unknown", cudaDev, gpuPath);
      return ncclInvalidUsage;
    }
  }

  if (ncclGdrCopy != NULL) {
    int forcePcie = 0;
    if (ncclGdrPinV2Available()) {
      if (ncclGdrIsInternalDmaBufBackend()) {
        forcePcie = 1;
      } else if (wrap_gdr_get_attribute(ncclGdrCopy, GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE, &forcePcie) !=
                 ncclSuccess) {
        forcePcie = 0;
      }
    }
    if (!forcePcie) {
      WARN("NET/MRC/RMA: Data Direct on GPU %d needs GDRCopy >= 2.5 with FORCE_PCIE support to map signals on PCIe; "
           "set NCCL_IB_DATA_DIRECT=0 otherwise",
           cudaDev);
      return ncclInvalidUsage;
    }
  }
  cComm->sigPinFlags = GDR_PIN_FLAG_FORCE_PCIE;
  INFO(NCCL_INIT | NCCL_NET, "NET/MRC/RMA: device %s uses Data Direct; signals are mapped on PCIe for GPU %d",
       mDev->devName, cudaDev);
  return ncclSuccess;
}

// Map a signal buffer for CPU stores. Failure leaves sigCpu NULL: the MR keeps its id, so ids stay
// consistent across ranks, and a signal that later targets it on this rank reports the error.
static void mrcRmaMapSignals(struct mrcRmaCollComm* cComm, struct mrcRmaMrHandle* mr, void* data, int type) {
  if (type == NCCL_PTR_HOST) {
    mr->sigCpu = (volatile uint64_t*)data;
    return;
  }
  if (ncclGdrCopy == NULL) {
    WARN("NET/MRC/RMA: cannot map signal buffer %p (%zu bytes): GDRCopy is unavailable", data, mr->bytes);
    return;
  }
  gdr_info_t info;
  if (ncclGdrCudaMapPointer(data, mr->bytes, &mr->gdrMh, &mr->gdrMap, &mr->gdrMapSize, &info, cComm->sigPinFlags) !=
      ncclSuccess) {
    WARN("NET/MRC/RMA: GDRCopy could not map signal buffer %p (%zu bytes)", data, mr->bytes);
    mr->gdrMap = NULL;
    return;
  }
  uintptr_t alignedAddr = (uintptr_t)data & GPU_PAGE_MASK;
  mr->sigCpu = (volatile uint64_t*)((char*)mr->gdrMap + ((uintptr_t)data - alignedAddr));
}

static void mrcRmaUnmapSignals(struct mrcRmaMrHandle* mr) {
  if (mr->gdrMap) (void)ncclGdrCudaUnmapPointer(mr->gdrMh, mr->gdrMap, mr->gdrMapSize);
  mr->gdrMap = NULL;
  mr->sigCpu = NULL;
  free(mr->shadow);
  mr->shadow = NULL;
}

// NCCL registers its signal buffers with NCCL_NET_MR_FLAG_FORCE_SO. Give each such buffer that is small
// enough to map an id. Registration is collective, so every rank assigns the same id; the caller checks it.
static int mrcRmaAssignSignalId(struct mrcRmaCollComm* cComm, struct mrcRmaMrHandle* mr, uint64_t mrFlags) {
  if (!(mrFlags & NCCL_NET_MR_FLAG_FORCE_SO)) return -1;
  if (mr->bytes == 0 || mr->bytes % sizeof(uint64_t) != 0 || mr->bytes > (size_t)ncclParamMrcRmaSignalMapMax())
    return -1;
  size_t slots = mr->bytes / sizeof(uint64_t);
  int slotBits = 0;
  while (((size_t)1 << slotBits) < slots) slotBits++;
  if (slotBits > MRC_RMA_SIG_MAX_SLOT_BITS) return -1;
  std::lock_guard<std::mutex> lock(cComm->sigMutex);
  for (int id = 0; id < MRC_RMA_SIG_BUFS; id++) {
    if (cComm->sigMrs[id].load(std::memory_order_relaxed) == NULL) {
      mr->slotBits = slotBits;
      cComm->sigMrs[id].store(mr, std::memory_order_release);
      return id;
    }
  }
  // Not fatal here: NCCL also registers FORCE_SO buffers it never signals (e.g. the host RMA flush buffer).
  mr->sigIdsExhausted = true;
  WARN("NET/MRC/RMA: all %d signal ids of this collective comm are in use; the %zu-byte buffer being registered "
       "cannot receive signals, and signals to it will fail. Reduce the number of RMA/GIN contexts or windows "
       "registered with NCCL_WIN_STRICT_ORDERING.",
       MRC_RMA_SIG_BUFS, mr->bytes);
  return -1;
}

static void mrcRmaReleaseSignalId(struct mrcRmaCollComm* cComm, struct mrcRmaMrHandle* mr) {
  if (mr->sigId < 0) return;
  std::lock_guard<std::mutex> lock(cComm->sigMutex);
  cComm->sigMrs[mr->sigId].store(NULL, std::memory_order_release);
  mr->sigId = -1;
}

static ncclResult_t mrcRmaDeregMrSym(void* collComm, void* mhandle);

static struct mrcRmaMrHandle* mrcRmaFindMr(struct mrcRmaCollComm* cComm, uint32_t mrId) {
  std::lock_guard<std::mutex> lock(cComm->sigMutex);
  auto it = cComm->mrById.find(mrId);
  return it == cComm->mrById.end() ? NULL : it->second;
}

struct mrcRmaMrExchange {
  uintptr_t va;
  uint32_t rkey;
  int32_t sigId;
  uint32_t mrId;
};

static ncclResult_t mrcRmaRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                         uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct mrcRmaCollComm* cComm = (struct mrcRmaCollComm*)collComm;
  struct mrcRmaMrExchange local, *all = NULL;
  struct mrcRmaMrHandle* mr = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mr, 1), ret, fail);
  mr->sigId = -1;
  mr->mrId = UINT32_MAX;
  mr->bytes = size;
  if (cComm->dataDirect && type == NCCL_PTR_CUDA && fd < 0) {
    // Without a DMA-BUF the NIC reaches this memory through its regular function, not the Data Direct path
    // signals are ordered with.
    WARN("NET/MRC/RMA: GPU memory %p must be registered through DMA-BUF on a Data Direct device; enable "
         "NCCL_DMABUF_ENABLE or set NCCL_IB_DATA_DIRECT=0",
         data);
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTONOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mrFlags,
                                                (void**)&mr->mrHandle),
                      ret, fail, NCCL_NET);
  NCCLCHECKGOTO(ncclCalloc(&mr->baseVas, cComm->nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&mr->rkeys, cComm->nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&all, cComm->nranks), ret, fail);

  mr->sigId = mrcRmaAssignSignalId(cComm, mr, mrFlags);
  if (mr->sigId >= 0) {
    mr->neverReset = (mrFlags & NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET) != 0;
    // Never-reset buffers are zeroed before registration and only ever written by us.
    if (mr->neverReset) NCCLCHECKGOTO(ncclCalloc(&mr->shadow, size / sizeof(uint64_t)), ret, fail);
    mrcRmaMapSignals(cComm, mr, data, type);
  }

  {
    std::lock_guard<std::mutex> lock(cComm->sigMutex);
    mr->mrId = cComm->nextMrId++;
    cComm->mrById[mr->mrId] = mr;
  }

  local.va = (uintptr_t)data;
  local.rkey = mr->mrHandle->mrs[0]->rkey;
  local.sigId = mr->sigId;
  local.mrId = mr->mrId;
  NCCLCHECKGOTO(mrcRmaAllGather(cComm, &local, all, sizeof(local)), ret, fail);
  for (int r = 0; r < cComm->nranks; r++) {
    mr->baseVas[r] = all[r].va;
    mr->rkeys[r] = all[r].rkey;
    if (all[r].sigId != mr->sigId || all[r].mrId != mr->mrId) {
      WARN("NET/MRC/RMA: symmetric registration mismatch: rank %d has signal id %d, region id %u; rank %d has %d, %u",
           cComm->rank, mr->sigId, mr->mrId, r, all[r].sigId, all[r].mrId);
      ret = ncclInternalError;
      goto fail;
    }
  }
  if (mr->sigId >= 0) {
    TRACE(NCCL_NET, "NET/MRC/RMA: signal buffer %p (%zu bytes, %s) has id %d, %d slot bits, cpu %p", data, size,
          mr->neverReset ? "never reset" : "resettable", mr->sigId, mr->slotBits, (void*)mr->sigCpu);
  }
  free(all);
  *mhandle = mr;
  return ncclSuccess;

fail:
  free(all);
  if (mr) {
    if (mr->mrHandle) (void)mrcRmaDeregMrSym(collComm, mr);
    else free(mr);
  }
  return ret;
}

static ncclResult_t mrcRmaRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                   void** mhandle) {
  return mrcRmaRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mrFlags, mhandle);
}

static ncclResult_t mrcRmaDeregMrSym(void* collComm, void* mhandle) {
  struct mrcRmaCollComm* cComm = (struct mrcRmaCollComm*)collComm;
  struct mrcRmaMrHandle* mr = (struct mrcRmaMrHandle*)mhandle;
  ncclResult_t ret = ncclSuccess;
  mrcRmaReleaseSignalId(cComm, mr);
  if (mr->mrId != UINT32_MAX) {
    std::lock_guard<std::mutex> lock(cComm->sigMutex);
    cComm->mrById.erase(mr->mrId);
  }
  mrcRmaUnmapSignals(mr);
  if (mr->mrHandle) NCCLCHECKIGNORE(ncclNetIb.deregMr(cComm->recvComm, mr->mrHandle), ret);
  free(mr->baseVas);
  free(mr->rkeys);
  free(mr);
  return ret;
}

/* ── Contexts ─────────────────────────────────────────────────────────── */

static ncclResult_t mrcRmaGetSendComm(struct mrcRmaCtx* ctx, int rank, struct ncclIbSendComm** comm) {
  *comm = (struct ncclIbSendComm*)ctx->fullSendComm[rank];
  if (*comm == NULL) {
    WARN("NET/MRC/RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

// Ring the doorbell for the pending chain, if any. Only the last WR of a chain is signaled; send completion
// is tracked per comm through ginSeq.
static ncclResult_t mrcRmaPostBatch(struct mrcRmaCtx* ctx) {
  struct mrcRmaWrBatch* batch = &ctx->batch;
  if (batch->nWrs == 0) return ncclSuccess;
  batch->wrs[batch->nWrs - 1].send_flags |= IBV_SEND_SIGNALED;
  batch->wrs[batch->nWrs - 1].next = NULL;
  struct ibv_send_wr* badWr;
  NCCLCHECK(wrap_mrc_post_send(batch->qp, &batch->wrs[0], &badWr));
  batch->nWrs = 0;
  batch->qp = NULL;
  return ncclSuccess;
}

static ncclResult_t mrcRmaReserveWrs(struct mrcRmaCtx* ctx, struct mrc_qp* qp, int nWrs, struct ibv_send_wr** wrs,
                                     struct ibv_sge** sges) {
  struct mrcRmaWrBatch* batch = &ctx->batch;
  if (batch->nWrs && (batch->qp != qp || batch->nWrs + nWrs > ctx->wrBatchsize)) NCCLCHECK(mrcRmaPostBatch(ctx));
  batch->qp = qp;
  *wrs = batch->wrs + batch->nWrs;
  *sges = batch->sges + batch->nWrs;
  memset(*wrs, 0, nWrs * sizeof(struct ibv_send_wr));
  memset(*sges, 0, nWrs * sizeof(struct ibv_sge));
  return ncclSuccess;
}

static ncclResult_t mrcRmaCommitWrs(struct mrcRmaCtx* ctx, int nWrs, bool aggregate) {
  struct mrcRmaWrBatch* batch = &ctx->batch;
  for (int i = 0; i < nWrs; i++) batch->wrs[batch->nWrs + i].next = batch->wrs + batch->nWrs + i + 1;
  batch->nWrs += nWrs;
  if (!aggregate || batch->nWrs + MRC_RMA_MAX_WRS_PER_OP > ctx->wrBatchsize) NCCLCHECK(mrcRmaPostBatch(ctx));
  return ncclSuccess;
}

// Reap send completions of the comm to `rank`. NCCL's requests advance ginSeq (only the last WR of a chain is
// signaled, so a completion covers every earlier request); get responses only count down getRespPending.
static ncclResult_t mrcRmaPollSend(struct mrcRmaCtx* ctx, int rank, struct ncclIbSendComm* comm, int max) {
  struct ibv_wc wc[MRC_RMA_POLL_MAX];
  int nDone = 0;
  NCCLCHECK(wrap_mrc_poll_cq(comm->devs[0].base.mrcCq, max, wc, &nDone));
  for (int i = 0; i < nDone; i++) {
    bool isGetResp = (wc[i].wr_id & MRC_RMA_WR_ID_GET_RESP) != 0;
    if (wc[i].status != IBV_WC_SUCCESS) {
      WARN("NET/MRC/RMA: %s completion error to rank %d: status=%d opcode=%d len=%u vendor err %u",
           isGetResp ? "get response" : "send", rank, wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err);
      printIbWcStatusHint(wc[i].status);
      return ncclRemoteError;
    }
    if (isGetResp) {
      ctx->getRespPending[rank]--;
      continue;
    }
    struct ncclIbRequest* wcReq = comm->base.reqs + wc[i].wr_id;
    if (wcReq->id > comm->ginSeq.completed) comm->ginSeq.completed = wcReq->id;
  }
  return ncclSuccess;
}

// Top the receive queue of `rank` back up to its depth with one chain of buffer-less WRs.
static ncclResult_t mrcRmaPostRecv(struct mrcRmaCtx* ctx, int rank) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ctx->fullRecvComm[rank];
  if (ctx->recvOutstanding[rank] >= ctx->recvDepth[rank]) return ncclSuccess;
  uint32_t needed = ctx->recvDepth[rank] - ctx->recvOutstanding[rank];
  struct ibv_recv_wr wrs[MRC_RMA_RECV_DEPTH_MAX];
  memset(wrs, 0, needed * sizeof(struct ibv_recv_wr));
  for (uint32_t i = 0; i < needed; i++) {
    wrs[i].wr_id = NCCL_IB_RECV_WR_ID_DUMMY;
    wrs[i].next = (i + 1 < needed) ? &wrs[i + 1] : NULL;
  }
  struct ibv_recv_wr* badWr;
  NCCLCHECK(wrap_mrc_post_recv(comm->base.qps[0].mrcQp, &wrs[0], &badWr));
  ctx->recvOutstanding[rank] += needed;
  return ncclSuccess;
}

static ncclResult_t mrcRmaDestroyContext(void* rmaCtx);

static ncclResult_t mrcRmaCreateContext(void* collComm, ncclRmaConfig_v17_t* config, void** rmaCtx) {
  ncclResult_t ret = ncclSuccess;
  struct mrcRmaCollComm* cComm = (struct mrcRmaCollComm*)collComm;
  int nranks = cComm->nranks;
  struct mrcRmaCtx* ctxs = NULL;
  void* lComm = NULL;
  char *handle = NULL, *handles = NULL;
  struct ncclIbMergedDev* mDev = &ncclIbMergedDevs[cComm->dev];
  *rmaCtx = NULL;

  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  int64_t wrBatchsize = ncclParamRmaIbWrBatchsize();
  if (wrBatchsize < MRC_RMA_MAX_WRS_PER_OP) wrBatchsize = MRC_RMA_MAX_WRS_PER_OP;
  if (wrBatchsize > MRC_RMA_WR_BATCH_MAX) wrBatchsize = MRC_RMA_WR_BATCH_MAX;

  NCCLCHECK(ncclCalloc(&ctxs, config->nContexts));
  for (int c = 0; c < config->nContexts; c++) {
    ctxs[c].cComm = cComm;
    ctxs[c].nContexts = config->nContexts;
    ctxs[c].nranks = nranks;
    ctxs[c].rank = cComm->rank;
    ctxs[c].wrBatchsize = (int)wrBatchsize;
  }

  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE * nranks), ret, fail);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE * cComm->rank;
  NCCLCHECKGOTO(ncclNetIb.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, fail);
  NCCLCHECKGOTO(mrcRmaAllGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);

  for (int c = 0; c < config->nContexts; c++) {
    struct mrcRmaCtx* gc = ctxs + c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void*) * nranks), ret, fail);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void*) * nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&gc->recvOutstanding, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&gc->recvDepth, nranks), ret, fail);

    // Connect to the requested peers, one round per rank distance with a barrier each, so ranks with
    // different peer counts stay in step. Self is always connected: a rank may target itself.
    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      bool wantConnect = (connectPeer == cComm->rank);
      bool wantAccept = (acceptPeer == cComm->rank);
      if (config->peerArrayCount > 0) {
        wantConnect |= (bool)bsearch(&connectPeer, config->peerArray, config->peerArrayCount, sizeof(int), compareInts);
        wantAccept |= (bool)bsearch(&acceptPeer, config->peerArray, config->peerArrayCount, sizeof(int), compareInts);
      }
      do {
        if (wantConnect && gc->fullSendComm[connectPeer] == NULL) {
          NCCLCHECKGOTO(ncclIbConnectImpl(cComm->ctx, cComm->dev, handles + NCCL_NET_HANDLE_MAXSIZE * connectPeer,
                                          &gc->fullSendComm[connectPeer], NULL, /*nQpsPerDev*/ 1,
                                          mrcRmaTrafficClass()),
                        ret, fail);
        }
        if (wantAccept && gc->fullRecvComm[acceptPeer] == NULL) {
          NCCLCHECKGOTO(ncclIbAcceptImpl(lComm, &gc->fullRecvComm[acceptPeer], NULL, /*nQpsPerDev*/ 1), ret, fail);
        }
        if (mrcRmaAborted(cComm->abortFlag)) {
          INFO(NCCL_NET, "NET/MRC/RMA: context connect aborted on rank %d/%d", cComm->rank, nranks);
          ret = ncclInternalError;
          goto fail;
        }
      } while ((wantConnect && gc->fullSendComm[connectPeer] == NULL) ||
               (wantAccept && gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(mrcRmaBarrier(cComm), ret, fail);
    }

    // Post receive WRs before any peer can send: a signal that finds none RNRs.
    for (int r = 0; r < nranks; r++) {
      struct ncclIbRecvComm* rc = (struct ncclIbRecvComm*)gc->fullRecvComm[r];
      if (rc == NULL) continue;
      gc->recvDepth[r] = MRC_RMA_RECV_DEPTH;
      if (rc->prepostReceiveWorkRequests) {
        uint32_t preposted = NET_IB_MAX_REQUESTS;
        if (rc->base.resiliency) {
          NCCLCHECKGOTO(ncclIbResiliencyDataRqSizeGet(rc->base.resiliency, rc->base.qps[0].devIndex, &preposted), ret,
                        fail);
        }
        gc->recvOutstanding[r] = preposted;
        gc->recvDepth[r] = preposted;
      }
      NCCLCHECKGOTO(mrcRmaPostRecv(gc, r), ret, fail);
    }

    // Data Direct writes of get data are not ordered with the CPU's completion, so NCCL's flush after a get
    // must read through the NIC (mrcRmaIFlush). Keep the loopback flush QP of the self comm available even
    // with NCCL_GDR_FLUSH_DISABLE=1.
    struct ncclIbRecvComm* selfRc = (struct ncclIbRecvComm*)gc->fullRecvComm[cComm->rank];
    if (cComm->dataDirect && selfRc && !selfRc->flushEnabled) {
      int tc = mrcRmaTrafficClass();
      selfRc->flushEnabled = 1;
      selfRc->flushQpSl = ncclParamIbSl() >= 0 ? (int)ncclParamIbSl() : 0;
      selfRc->flushQpTc = tc >= 0 ? tc : 0;
    }
  }

  // Get descriptors: one MR for all contexts, registered collectively so every rank knows every inbox.
  {
    int64_t getSlots = ncclParamMrcRmaGetSlots();
    if (getSlots < 0) getSlots = 0;
    if (getSlots > MRC_RMA_CTRL_ARG_MASK + 1) getSlots = MRC_RMA_CTRL_ARG_MASK + 1;
    int* allSlots = NULL;
    NCCLCHECKGOTO(ncclCalloc(&allSlots, nranks), ret, fail);
    int mySlots = (int)getSlots;
    ret = mrcRmaAllGather(cComm, &mySlots, allSlots, sizeof(int));
    for (int r = 0; ret == ncclSuccess && r < nranks; r++) {
      if (allSlots[r] != mySlots) {
        WARN("NET/MRC/RMA: NCCL_MRC_RMA_GET_SLOTS differs across ranks (rank %d: %d, rank %d: %d)", cComm->rank,
             mySlots, r, allSlots[r]);
        ret = ncclInvalidUsage;
      }
    }
    free(allSlots);
    if (ret != ncclSuccess) goto fail;
    if (getSlots > 0) {
      size_t perCtx = 2 * (size_t)nranks * getSlots * sizeof(struct mrcRmaGetDesc);
      char* getHost = NULL;
      NCCLCHECKGOTO(ncclIbMalloc((void**)&getHost, perCtx * config->nContexts), ret, fail);
      memset(getHost, 0, perCtx * config->nContexts);
      ctxs[0].getBuf = getHost;  // owned by context 0 from here, freed on destroy
      NCCLCHECKGOTO(mrcRmaRegMrSym(cComm, getHost, perCtx * config->nContexts, NCCL_PTR_HOST, 0,
                                   (void**)&ctxs[0].getMr),
                    ret, fail);
      for (int c = 0; c < config->nContexts; c++) {
        struct mrcRmaCtx* gc = ctxs + c;
        gc->getSlots = (int)getSlots;
        gc->getMr = ctxs[0].getMr;
        gc->getBufOff = perCtx * c;
        gc->getBuf = getHost + gc->getBufOff;
        NCCLCHECKGOTO(ncclCalloc(&gc->getSlotReq, (size_t)nranks * getSlots), ret, fail);
        for (size_t i = 0; i < (size_t)nranks * getSlots; i++) gc->getSlotReq[i] = MRC_RMA_GET_SLOT_FREE;
        NCCLCHECKGOTO(ncclCalloc(&gc->getCursor, nranks), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(&gc->getDone, (size_t)nranks * NET_IB_MAX_REQUESTS), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(&gc->getRespPending, nranks), ret, fail);
        gc->getQueue = new (std::nothrow) std::deque<struct mrcRmaQueuedGet>[nranks];
        gc->getServeQueue = new (std::nothrow) std::deque<uint32_t>[nranks];
        if (gc->getQueue == NULL || gc->getServeQueue == NULL) {
          ret = ncclSystemError;
          goto fail;
        }
      }
    }
  }

  // Every rank has its receive WRs posted before anyone returns and starts sending.
  NCCLCHECKGOTO(mrcRmaBarrier(cComm), ret, fail);

  INFO(NCCL_INIT | NCCL_NET, "NET/MRC/RMA: rank %d created %d contexts over %d ranks (%d peers) on %s",
       cComm->rank, config->nContexts, nranks, config->peerArrayCount, mDev->devName);
  *rmaCtx = ctxs;

exit:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);
  return ret;
fail:
  (void)mrcRmaDestroyContext(ctxs);
  goto exit;
}

// Wait for every send posted on this context to complete, so closing the send QPs does not drop signals still
// in flight. Bounded by a timeout so a dead peer cannot wedge teardown.
static void mrcRmaDrainSends(struct mrcRmaCtx* gc, int context) {
  if (gc->fullSendComm == NULL) return;
  struct timespec t0, tnow;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < gc->nranks; i++) {
    struct ncclIbSendComm* sc = (struct ncclIbSendComm*)gc->fullSendComm[i];
    if (sc == NULL) continue;
    while (sc->ginSeq.completed < sc->ginSeq.posted || (gc->getRespPending && gc->getRespPending[i] > 0)) {
      if (mrcRmaPollSend(gc, i, sc, MRC_RMA_POLL_MAX) != ncclSuccess) {
        WARN("NET/MRC/RMA: send completion error to rank %d ctx=%d during teardown", i, context);
        return;
      }
      clock_gettime(CLOCK_MONOTONIC, &tnow);
      double elapsed = (double)(tnow.tv_sec - t0.tv_sec) + (double)(tnow.tv_nsec - t0.tv_nsec) / 1e9;
      if (elapsed > 10.0) {
        WARN("NET/MRC/RMA: timed out draining %lu outstanding sends and %u get responses to rank %d ctx=%d",
             (unsigned long)(sc->ginSeq.posted - sc->ginSeq.completed),
             gc->getRespPending ? gc->getRespPending[i] : 0, i, context);
        return;
      }
    }
  }
}

static ncclResult_t mrcRmaDestroyContext(void* rmaCtx) {
  struct mrcRmaCtx* ctxs = (struct mrcRmaCtx*)rmaCtx;
  if (ctxs == NULL) return ncclSuccess;
  ncclResult_t ret = ncclSuccess;
  int nContexts = ctxs[0].nContexts;
  int nranks = ctxs[0].nranks;
  for (int c = 0; c < nContexts; c++) {
    struct mrcRmaCtx* gc = ctxs + c;
    if (gc->batch.nWrs) NCCLCHECKIGNORE(mrcRmaPostBatch(gc), ret);
    mrcRmaDrainSends(gc, c);
    if (gc->fullRecvComm) {
      for (int i = 0; i < nranks; i++) {
        if (gc->fullRecvComm[i]) NCCLCHECKIGNORE(ncclNetIb.closeRecv(gc->fullRecvComm[i]), ret);
      }
      free(gc->fullRecvComm);
    }
    if (gc->fullSendComm) {
      for (int i = 0; i < nranks; i++) {
        if (gc->fullSendComm[i]) NCCLCHECKIGNORE(ncclNetIb.closeSend(gc->fullSendComm[i]), ret);
      }
      free(gc->fullSendComm);
    }
    free(gc->recvOutstanding);
    free(gc->recvDepth);
    free(gc->getSlotReq);
    free(gc->getCursor);
    free(gc->getDone);
    free(gc->getRespPending);
    delete[] gc->getQueue;
    delete[] gc->getServeQueue;
  }
  if (ctxs[0].getMr) NCCLCHECKIGNORE(mrcRmaDeregMrSym(ctxs[0].cComm, ctxs[0].getMr), ret);
  free(ctxs[0].getBuf);  // context 0's region starts the allocation
  free(ctxs);
  return ret;
}

/* ── Data path ────────────────────────────────────────────────────────── */

static ncclResult_t mrcRmaNewRequest(struct mrcRmaCtx* ctx, struct ncclIbSendComm* comm, uint32_t rank,
                                     struct ncclIbRequest** reqPtr) {
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = ctx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->id = ++comm->ginSeq.posted;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) req->devBases[i] = &comm->devs[i].base;
  *reqPtr = req;
  return ncclSuccess;
}

static ncclResult_t mrcRmaIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                               uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
  struct mrcRmaCtx* ctx = &((struct mrcRmaCtx*)rmaCtx)[context];
  struct mrcRmaMrHandle* srcMr = (struct mrcRmaMrHandle*)srcMhandle;
  struct mrcRmaMrHandle* dstMr = (struct mrcRmaMrHandle*)dstMhandle;
  struct ncclIbSendComm* comm;
  NCCLCHECK(mrcRmaGetSendComm(ctx, rank, &comm));
  struct ncclIbRequest* req;
  NCCLCHECK(mrcRmaNewRequest(ctx, comm, rank, &req));

  struct ibv_send_wr* wr;
  struct ibv_sge* sge;
  NCCLCHECK(mrcRmaReserveWrs(ctx, comm->base.qps[0].mrcQp, 1, &wr, &sge));
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->wr_id = req - comm->base.reqs;
  wr->wr.rdma.remote_addr = (uint64_t)(dstMr->baseVas[rank] + dstOff);
  wr->wr.rdma.rkey = dstMr->rkeys[rank];
  wr->sg_list = sge;
  wr->num_sge = 1;
  sge->addr = (uintptr_t)(srcMr->baseVas[ctx->rank] + srcOff);
  sge->length = size;
  sge->lkey = srcMr->mrHandle->mrs[0]->lkey;
  NCCLCHECK(mrcRmaCommitWrs(ctx, 1, optFlags & ncclRmaOptFlagsAggregateRequests));
  *request = req;
  return ncclSuccess;
}

static ncclResult_t mrcRmaIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                     uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                     void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                     uint32_t optFlags, void** request) {
  // Strong and weak signals are the same here: a signal rides the same QP as every earlier put to that peer.
  (void)isStrongSignal;
  struct mrcRmaCtx* ctx = &((struct mrcRmaCtx*)rmaCtx)[context];
  struct mrcRmaMrHandle* srcMr = (struct mrcRmaMrHandle*)srcMhandle;
  struct mrcRmaMrHandle* dstMr = (struct mrcRmaMrHandle*)dstMhandle;
  struct mrcRmaMrHandle* sigMr = (struct mrcRmaMrHandle*)signalMhandle;

  // Validate before taking a request, so a rejected op leaves no trace in ginSeq.
  if (sigMr != NULL && sigMr->sigIdsExhausted) {
    WARN("NET/MRC/RMA: signal target buffer has no signal id: all %d ids of its collective comm were in use when "
         "it was registered", MRC_RMA_SIG_BUFS);
    return ncclInvalidUsage;
  }
  if (sigMr == NULL || sigMr->sigId < 0) {
    WARN("NET/MRC/RMA: signal target is not a registered signal buffer (MRC delivers signals only to buffers "
         "registered with NCCL_NET_MR_FLAG_FORCE_SO of at most NCCL_MRC_RMA_SIGNAL_MAP_MAX bytes; VA signals are "
         "not supported)");
    return ncclInvalidUsage;
  }
  uint64_t value;
  if (signalOp == NCCL_NET_SIGNAL_OP_INC) {
    value = 1;
  } else if (signalOp == NCCL_NET_SIGNAL_OP_ADD) {
    value = signalValue;
  } else {
    WARN("NET/MRC/RMA: unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }
  if (signalOff % sizeof(uint64_t) != 0 || signalOff >= sigMr->bytes) {
    WARN("NET/MRC/RMA: invalid signal offset %lu (signal buffer size %zu)", (unsigned long)signalOff, sigMr->bytes);
    return ncclInvalidUsage;
  }
  const int mantBits = MRC_RMA_SIG_FIELD_BITS - sigMr->slotBits;
  const uint64_t mantMask = (1ull << mantBits) - 1;
  uint32_t head = ((uint32_t)sigMr->sigId << MRC_RMA_SIG_ID_SHIFT) |
                  ((uint32_t)(signalOff / sizeof(uint64_t)) << (mantBits + MRC_RMA_SIG_EXP_BITS));
  uint32_t imms[MRC_RMA_SIG_MAX_DIGITS];
  int nImms = 0;
  for (int digit = 0; value != 0; digit++, value >>= mantBits) {
    if (digit >= MRC_RMA_SIG_MAX_DIGITS) {
      WARN("NET/MRC/RMA: signal add value %lu exceeds the maximum MRC can carry for this buffer (%d bits)",
           (unsigned long)signalValue, mantBits * MRC_RMA_SIG_MAX_DIGITS);
      return ncclInvalidArgument;
    }
    uint32_t mant = (uint32_t)(value & mantMask);
    if (mant) imms[nImms++] = head | ((uint32_t)digit << mantBits) | mant;
  }
  if (nImms == 0) imms[nImms++] = head;  // add 0: still a message, for the put and the completion

  struct ncclIbSendComm* comm;
  NCCLCHECK(mrcRmaGetSendComm(ctx, rank, &comm));
  struct ncclIbRequest* req;
  NCCLCHECK(mrcRmaNewRequest(ctx, comm, rank, &req));

  struct ibv_send_wr* wr;
  struct ibv_sge* sge;
  NCCLCHECK(mrcRmaReserveWrs(ctx, comm->base.qps[0].mrcQp, nImms, &wr, &sge));
  for (int i = 0; i < nImms; i++) {
    wr[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr[i].wr_id = req - comm->base.reqs;
    wr[i].imm_data = htobe32(imms[i]);
    // Zero-length: the target address is not accessed, but must be valid under the rkey.
    wr[i].wr.rdma.remote_addr = (uint64_t)(sigMr->baseVas[rank] + signalOff);
    wr[i].wr.rdma.rkey = sigMr->rkeys[rank];
  }
  // The payload rides the last immediate, so the full value is never visible before the data.
  if (size > 0 && dstMr) {
    struct ibv_send_wr* last = &wr[nImms - 1];
    last->wr.rdma.remote_addr = (uint64_t)(dstMr->baseVas[rank] + dstOff);
    last->wr.rdma.rkey = dstMr->rkeys[rank];
    last->sg_list = &sge[nImms - 1];
    last->num_sge = 1;
    sge[nImms - 1].addr = (uintptr_t)(srcMr->baseVas[ctx->rank] + srcOff);
    sge[nImms - 1].length = size;
    sge[nImms - 1].lkey = srcMr->mrHandle->mrs[0]->lkey;
  }
  NCCLCHECK(mrcRmaCommitWrs(ctx, nImms, optFlags & ncclRmaOptFlagsAggregateRequests));
  *request = req;
  return ncclSuccess;
}

// A free descriptor slot for a get to `rank`, or -1.
static int mrcRmaFreeGetSlot(struct mrcRmaCtx* ctx, int rank) {
  uint16_t* slots = ctx->getSlotReq + (size_t)rank * ctx->getSlots;
  for (int i = 0; i < ctx->getSlots; i++) {
    int s = (ctx->getCursor[rank] + i) % ctx->getSlots;
    if (slots[s] == MRC_RMA_GET_SLOT_FREE) {
      ctx->getCursor[rank] = (s + 1) % ctx->getSlots;
      return s;
    }
  }
  return -1;
}

// Send the descriptor of request reqIdx to `rank` through `slot`. The request gets its ginSeq id here, when its
// WR enters the QP, so the ids keep QP order even for gets that waited for a slot.
static ncclResult_t mrcRmaPostGet(struct mrcRmaCtx* ctx, int rank, int reqIdx, int slot,
                                  const struct mrcRmaGetDesc* getDesc, bool aggregate) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ctx->fullSendComm[rank];
  comm->base.reqs[reqIdx].id = ++comm->ginSeq.posted;
  ctx->getSlotReq[(size_t)rank * ctx->getSlots + slot] = (uint16_t)reqIdx;
  // The outbox slot is not rewritten before the response, which the target sends after reading its inbox copy.
  struct mrcRmaGetDesc* desc = (struct mrcRmaGetDesc*)(ctx->getBuf + mrcRmaGetOutboxOff(ctx, rank, slot));
  *desc = *getDesc;

  struct ibv_send_wr* wr;
  struct ibv_sge* sge;
  NCCLCHECK(mrcRmaReserveWrs(ctx, comm->base.qps[0].mrcQp, 1, &wr, &sge));
  wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr->wr_id = reqIdx;
  wr->imm_data = htobe32(MRC_RMA_CTRL_IMM(MRC_RMA_CTRL_GET_REQ, slot));
  wr->wr.rdma.remote_addr =
    (uint64_t)(ctx->getMr->baseVas[rank] + ctx->getBufOff + mrcRmaGetInboxOff(ctx, ctx->rank, slot));
  wr->wr.rdma.rkey = ctx->getMr->rkeys[rank];
  wr->sg_list = sge;
  wr->num_sge = 1;
  sge->addr = (uintptr_t)desc;
  sge->length = sizeof(*desc);
  sge->lkey = ctx->getMr->mrHandle->mrs[0]->lkey;
  return mrcRmaCommitWrs(ctx, 1, aggregate);
}

/* Get over MRC, which has no RDMA Read, is a receiver-driven write:
 *  1. the requester writes a descriptor (source and destination region ids, offsets, size) into its slot of the
 *     target's inbox, with a Write-with-Immediate carrying GET_REQ and the slot;
 *  2. the target's progress reads the descriptor and writes the data into the requester's memory with one
 *     Write-with-Immediate carrying GET_RESP and the slot (mrcRmaServeGet);
 *  3. the requester's progress reaps that completion, MRC's sign that the data is placed, and the get is done
 *     (mrcRmaCompleteGet). NCCL then flushes (mrcRmaIFlush) before the GPU reads the data.
 * When all slots to the target wait on responses, the get is queued rather than waited for: the proxy thread
 * must keep progressing its other comms, whose peers may be the ones serving us.
 */
static ncclResult_t mrcRmaIGet(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                               uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags,
                               void** request) {
  struct mrcRmaCtx* ctx = &((struct mrcRmaCtx*)rmaCtx)[context];
  struct mrcRmaMrHandle* remoteMr = (struct mrcRmaMrHandle*)remoteMhandle;
  struct mrcRmaMrHandle* localMr = (struct mrcRmaMrHandle*)localMhandle;
  if (ctx->getSlots == 0) {
    WARN("NET/MRC/RMA: get is disabled (NCCL_MRC_RMA_GET_SLOTS=0)");
    return ncclInvalidUsage;
  }
  if (remoteMr == NULL || localMr == NULL || size > MRC_RMA_GET_MAX_BYTES || remoteOff + size > remoteMr->bytes ||
      localOff + size > localMr->bytes) {
    WARN("NET/MRC/RMA: invalid get of %zu bytes from offset %lu to offset %lu (region sizes %zu, %zu)", size,
         (unsigned long)remoteOff, (unsigned long)localOff, remoteMr ? remoteMr->bytes : 0,
         localMr ? localMr->bytes : 0);
    return ncclInvalidArgument;
  }
  if (ctx->fullRecvComm[rank] == NULL) {
    WARN("NET/MRC/RMA: get from rank %d, which is not connected to send back to rank %d", rank, ctx->rank);
    return ncclInvalidUsage;
  }
  struct ncclIbSendComm* comm;
  NCCLCHECK(mrcRmaGetSendComm(ctx, rank, &comm));
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = ctx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  req->id = 0;  // assigned when posted
  for (int i = 0; i < comm->base.vProps.ndevs; i++) req->devBases[i] = &comm->devs[i].base;
  int reqIdx = req - comm->base.reqs;
  ctx->getDone[(size_t)rank * NET_IB_MAX_REQUESTS + reqIdx] = 0;

  struct mrcRmaGetDesc desc = {remoteMr->mrId, localMr->mrId, remoteOff, localOff, size};
  int slot = ctx->getQueue[rank].empty() ? mrcRmaFreeGetSlot(ctx, rank) : -1;
  if (slot >= 0) {
    NCCLCHECK(mrcRmaPostGet(ctx, rank, reqIdx, slot, &desc, optFlags & ncclRmaOptFlagsAggregateRequests));
  } else {
    ctx->getQueue[rank].push_back({(uint16_t)reqIdx, desc});
  }
  *request = req;
  return ncclSuccess;
}

// Make completed gets visible to the GPU: an RDMA Read of the local buffer over net_ib's loopback flush QP
// (a regular RC QP), which returns only after the NIC's earlier writes on that path have landed. Without a
// flush QP (NCCL_GDR_FLUSH_DISABLE=1 and no Data Direct) there is nothing to do.
static ncclResult_t mrcRmaIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  (void)rank;  // the flush is local whatever queue NCCL posted it on
  struct mrcRmaCtx* ctx = &((struct mrcRmaCtx*)rmaCtx)[context];
  struct mrcRmaMrHandle* mr = (struct mrcRmaMrHandle*)mhandle;
  *request = NULL;
  struct ncclIbRecvComm* rc = (struct ncclIbRecvComm*)ctx->fullRecvComm[ctx->rank];
  if (rc == NULL || mr == NULL) {
    WARN("NET/MRC/RMA: flush needs the self connection and a registered buffer");
    return ncclInvalidUsage;
  }
  if (!rc->flushEnabled) return ncclSuccess;
  NCCLCHECK(mrcRmaPostBatch(ctx));
  void* data = (void*)mr->baseVas[ctx->rank];
  int sizes = 1;
  void* mrHandle = mr->mrHandle;
  return ncclNetIb.iflush(rc, 1, &data, &sizes, &mrHandle, request);
}

static ncclResult_t mrcRmaPollRecv(struct mrcRmaCtx* ctx, int rank, int pollCount, bool* stored);

static ncclResult_t mrcRmaTest(void* collComm, void* request, int* done) {
  (void)collComm;
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) return ncclNetIb.test(req, done, NULL);
  struct mrcRmaCtx* ctx = (struct mrcRmaCtx*)req->rmaProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  // The caller is waiting on this op, so a deferred doorbell must not hold it back.
  NCCLCHECK(mrcRmaPostBatch(ctx));

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ctx->fullSendComm[rank];
  if (req->id > comm->ginSeq.completed) NCCLCHECK(mrcRmaPollSend(ctx, rank, comm, 4));
  if (req->id > comm->ginSeq.completed) return ncclSuccess;
  if (req->type == NCCL_NET_IB_REQ_GIN_IGET) {
    // A get is done once the data is back, which the response from the target tells. A queued get has id 0
    // until posted, so it reaches this check and waits here.
    uint8_t* responded = &ctx->getDone[(size_t)rank * NET_IB_MAX_REQUESTS + (req - comm->base.reqs)];
    if (!*responded) {
      bool stored = false;
      ncclResult_t ret = mrcRmaPollRecv(ctx, rank, MRC_RMA_POLL_MAX, &stored);
      if (stored) wc_store_fence();
      NCCLCHECK(ret);
      if (!*responded) return ncclSuccess;
    }
    *responded = 0;
  }
  *done = 1;
  NCCLCHECK(ncclIbFreeRequest(req));
  return ncclSuccess;
}

/* ── Signal delivery ──────────────────────────────────────────────────── */

// Write the data described in `requester`'s inbox slot back into its memory, with GET_RESP as the immediate.
static ncclResult_t mrcRmaPostGetResp(struct mrcRmaCtx* ctx, int requester, uint32_t slot) {
  struct mrcRmaGetDesc desc;
  std::atomic_thread_fence(std::memory_order_acquire);
  memcpy(&desc, ctx->getBuf + mrcRmaGetInboxOff(ctx, requester, slot), sizeof(desc));
  struct mrcRmaMrHandle* srcMr = mrcRmaFindMr(ctx->cComm, desc.srcMrId);
  struct mrcRmaMrHandle* dstMr = mrcRmaFindMr(ctx->cComm, desc.dstMrId);
  if (srcMr == NULL || dstMr == NULL || desc.size > MRC_RMA_GET_MAX_BYTES || desc.srcOff + desc.size > srcMr->bytes ||
      desc.dstOff + desc.size > dstMr->bytes) {
    WARN("NET/MRC/RMA: invalid get request from rank %d: %lu bytes from region %u offset %lu to region %u offset %lu "
         "(%s)", requester, (unsigned long)desc.size, desc.srcMrId, (unsigned long)desc.srcOff, desc.dstMrId,
         (unsigned long)desc.dstOff, srcMr && dstMr ? "out of bounds" : "region not registered on this rank");
    return ncclInternalError;
  }
  struct ncclIbSendComm* comm;
  NCCLCHECK(mrcRmaGetSendComm(ctx, requester, &comm));
  // Keep issue order on the QP, and keep this WR out of NCCL's chains: it holds no ncclIbRequest.
  NCCLCHECK(mrcRmaPostBatch(ctx));
  struct ibv_send_wr wr;
  struct ibv_sge sge;
  memset(&wr, 0, sizeof(wr));
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = MRC_RMA_WR_ID_GET_RESP;
  wr.imm_data = htobe32(MRC_RMA_CTRL_IMM(MRC_RMA_CTRL_GET_RESP, slot));
  wr.wr.rdma.remote_addr = (uint64_t)(dstMr->baseVas[requester] + desc.dstOff);
  wr.wr.rdma.rkey = dstMr->rkeys[requester];
  if (desc.size > 0) {
    sge.addr = (uintptr_t)(srcMr->baseVas[ctx->rank] + desc.srcOff);
    sge.length = (uint32_t)desc.size;
    sge.lkey = srcMr->mrHandle->mrs[0]->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  struct ibv_send_wr* badWr;
  NCCLCHECK(wrap_mrc_post_send(comm->base.qps[0].mrcQp, &wr, &badWr));
  ctx->getRespPending[requester]++;
  return ncclSuccess;
}

// Serve a get request from `requester`, or defer it while MRC_RMA_GET_RESP_MAX responses are in flight. The
// requester does not reuse the slot before its response, so the descriptor stays valid until then.
static ncclResult_t mrcRmaServeGet(struct mrcRmaCtx* ctx, int requester, uint32_t slot) {
  if (slot >= (uint32_t)ctx->getSlots) {
    WARN("NET/MRC/RMA: get request from rank %d for slot %u, but %d slots are configured", requester, slot,
         ctx->getSlots);
    return ncclInternalError;
  }
  std::deque<uint32_t>& deferred = ctx->getServeQueue[requester];
  if (!deferred.empty() || ctx->getRespPending[requester] >= MRC_RMA_GET_RESP_MAX) {
    deferred.push_back(slot);
    return ncclSuccess;
  }
  return mrcRmaPostGetResp(ctx, requester, slot);
}

// Serve deferred get requests from `requester` as response credit allows.
static ncclResult_t mrcRmaServeDeferredGets(struct mrcRmaCtx* ctx, int requester) {
  std::deque<uint32_t>& deferred = ctx->getServeQueue[requester];
  while (!deferred.empty() && ctx->getRespPending[requester] < MRC_RMA_GET_RESP_MAX) {
    uint32_t slot = deferred.front();
    deferred.pop_front();
    NCCLCHECK(mrcRmaPostGetResp(ctx, requester, slot));
  }
  return ncclSuccess;
}

// The data of the get in `slot` to `target` is placed: complete the request waiting on it and free the slot.
static ncclResult_t mrcRmaCompleteGet(struct mrcRmaCtx* ctx, int target, uint32_t slot) {
  uint16_t* slotReq = slot < (uint32_t)ctx->getSlots ? &ctx->getSlotReq[(size_t)target * ctx->getSlots + slot] : NULL;
  if (slotReq == NULL || *slotReq == MRC_RMA_GET_SLOT_FREE) {
    WARN("NET/MRC/RMA: unexpected get response from rank %d for slot %u", target, slot);
    return ncclInternalError;
  }
  ctx->getDone[(size_t)target * NET_IB_MAX_REQUESTS + *slotReq] = 1;
  *slotReq = MRC_RMA_GET_SLOT_FREE;
  if (!ctx->getQueue[target].empty()) {
    struct mrcRmaQueuedGet next = ctx->getQueue[target].front();
    ctx->getQueue[target].pop_front();
    NCCLCHECK(mrcRmaPostGet(ctx, target, next.reqIdx, slot, &next.desc, /*aggregate=*/false));
  }
  return ncclSuccess;
}

static ncclResult_t mrcRmaHandleRecv(struct mrcRmaCtx* ctx, int rank, struct ibv_wc* wc, bool* stored) {
  // Refill before processing, so reposting isn't delayed behind the per-completion work.
  ctx->recvOutstanding[rank] -= 1;
  if (ctx->recvDepth[rank] - ctx->recvOutstanding[rank] >= MRC_RMA_RECV_REFILL) NCCLCHECK(mrcRmaPostRecv(ctx, rank));

  if (wc->status != IBV_WC_SUCCESS) {
    WARN("NET/MRC/RMA: receive completion error from rank %d: status=%d opcode=%d vendor err %u", rank, wc->status,
         wc->opcode, wc->vendor_err);
    return ncclRemoteError;
  }
  if (wc->opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    WARN("NET/MRC/RMA: unexpected receive completion opcode %d from rank %d", wc->opcode, rank);
    return ncclInternalError;
  }
  uint32_t imm = be32toh(wc->imm_data);
  int id = imm >> MRC_RMA_SIG_ID_SHIFT;
  if (id == MRC_RMA_CTRL_ID) {
    uint32_t op = (imm >> MRC_RMA_CTRL_OP_SHIFT) & 3, arg = imm & MRC_RMA_CTRL_ARG_MASK;
    if (op == MRC_RMA_CTRL_GET_REQ) return mrcRmaServeGet(ctx, rank, arg);
    if (op == MRC_RMA_CTRL_GET_RESP) return mrcRmaCompleteGet(ctx, rank, arg);
    WARN("NET/MRC/RMA: unknown control message 0x%08x from rank %d", imm, rank);
    return ncclInternalError;
  }
  struct mrcRmaMrHandle* mr = ctx->cComm->sigMrs[id].load(std::memory_order_acquire);
  if (mr == NULL || mr->sigCpu == NULL) {
    WARN("NET/MRC/RMA: signal from rank %d for signal buffer id %d, which is %s on this rank", rank, id,
         mr ? "not mapped" : "not registered");
    return ncclInternalError;
  }
  const int mantBits = MRC_RMA_SIG_FIELD_BITS - mr->slotBits;
  uint32_t mant = imm & ((1u << mantBits) - 1);
  uint32_t digit = (imm >> mantBits) & (MRC_RMA_SIG_MAX_DIGITS - 1);
  uint32_t slot = (imm >> (mantBits + MRC_RMA_SIG_EXP_BITS)) & ((1u << mr->slotBits) - 1);
  if (slot >= mr->bytes / sizeof(uint64_t) || digit * mantBits >= 64) {
    WARN("NET/MRC/RMA: malformed signal 0x%08x from rank %d (slot %u, digit %u)", imm, rank, slot, digit);
    return ncclInternalError;
  }
  uint64_t add = (uint64_t)mant << (digit * mantBits);
  uint64_t v;
  if (mr->shadow) {
    // Never reset by NCCL, so the host copy is current and the slot never needs to be read back.
    v = (mr->shadow[slot] += add);
  } else {
    v = COMPILER_ATOMIC_LOAD(&mr->sigCpu[slot], std::memory_order_relaxed) + add;
  }
  COMPILER_ATOMIC_STORE(&mr->sigCpu[slot], v, std::memory_order_relaxed);
  *stored = true;
  return ncclSuccess;
}

// Reap and handle up to pollCount receive completions from `rank`. Sets *stored when a signal was written.
static ncclResult_t mrcRmaPollRecv(struct mrcRmaCtx* ctx, int rank, int pollCount, bool* stored) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ctx->fullRecvComm[rank];
  if (comm == NULL) return ncclSuccess;
  struct ibv_wc wc[MRC_RMA_POLL_MAX];
  int wrDone = 0;
  NCCLCHECK(wrap_mrc_poll_cq(comm->devs[0].base.mrcCq, pollCount, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) NCCLCHECK(mrcRmaHandleRecv(ctx, rank, &wc[i], stored));
  return ncclSuccess;
}

static ncclResult_t mrcRmaProgress(void* rmaCtx) {
  struct mrcRmaCtx* ctxs = (struct mrcRmaCtx*)rmaCtx;
  if (ctxs[0].asyncError != ncclSuccess) return ctxs[0].asyncError;
  int pollCount = ncclParamMrcRmaPollCount();
  if (pollCount < 1) pollCount = 1;
  if (pollCount > MRC_RMA_POLL_MAX) pollCount = MRC_RMA_POLL_MAX;
  bool stored = false;
  ncclResult_t ret = ncclSuccess;

  for (int c = 0; c < ctxs[0].nContexts; c++) {
    struct mrcRmaCtx* ctx = ctxs + c;
    for (int rank = 0; rank < ctx->nranks; rank++) {
      NCCLCHECKGOTO(mrcRmaPollRecv(ctx, rank, pollCount, &stored), ret, exit);
      // Get responses we posted are reaped here, as nobody may be testing requests on that comm, and deferred
      // requests are served once credit allows (test() may have reaped the responses already).
      if (ctx->getRespPending && (ctx->getRespPending[rank] > 0 || !ctx->getServeQueue[rank].empty())) {
        NCCLCHECKGOTO(mrcRmaPollSend(ctx, rank, (struct ncclIbSendComm*)ctx->fullSendComm[rank], pollCount), ret,
                      exit);
        NCCLCHECKGOTO(mrcRmaServeDeferredGets(ctx, rank), ret, exit);
      }
    }
  }

exit:
  // Flush the write-combined signal stores so waiters observe them.
  if (stored) wc_store_fence();
  if (ret != ncclSuccess) ctxs[0].asyncError = ret;
  return ret;
}

static ncclResult_t mrcRmaQueryLastError(void* rmaCtx, bool* hasError) {
  struct mrcRmaCtx* ctxs = (struct mrcRmaCtx*)rmaCtx;
  *hasError = ctxs != NULL && ctxs[0].asyncError != ncclSuccess;
  return ncclSuccess;
}

static ncclResult_t mrcRmaDevices(int* ndev) { return ncclIbDevices(ndev); }

static ncclResult_t mrcRmaListen(void* ctx, int dev, void* handle, void** listenComm) {
  return ncclIbListen(ctx, dev, handle, listenComm);
}

static ncclResult_t mrcRmaCloseListen(void* listenComm) { return ncclIbCloseListen(listenComm); }

extern "C" {
__attribute__((visibility("default"))) ncclRma_v17_t ncclRmaPlugin_v17 = {
  "MRC",
  mrcRmaInit,
  mrcRmaDevices,
  mrcRmaGetRmaProperties,
  mrcRmaGetProperties,
  mrcRmaListen,
  mrcRmaConnect,
  mrcRmaCreateContext,
  mrcRmaRegMrSym,
  mrcRmaRegMrSymDmaBuf,
  mrcRmaDeregMrSym,
  mrcRmaDestroyContext,
  mrcRmaCloseColl,
  mrcRmaCloseListen,
  mrcRmaIPut,
  mrcRmaIPutSignal,
  mrcRmaIGet,
  mrcRmaIFlush,
  mrcRmaTest,
  mrcRmaProgress,
  mrcRmaQueryLastError,
  mrcRmaFinalize,
};
}
