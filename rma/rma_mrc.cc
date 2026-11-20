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
// signal buffer.

#include <atomic>
#include <limits.h>
#include <mutex>
#include <new>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
NCCL_PARAM(RmaIbWrBatchsize, "RMA_IB_WR_BATCHSIZE", 64);
NCCL_PARAM(MrcRmaPollCount, "MRC_RMA_POLL_COUNT", 16);
NCCL_PARAM(MrcRmaSignalMapMax, "MRC_RMA_SIGNAL_MAP_MAX", 64 << 20);

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
  std::mutex sigMutex;
  std::atomic<struct mrcRmaMrHandle*> sigMrs[MRC_RMA_SIG_IDS];
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
};

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
  for (int id = 0; id < MRC_RMA_SIG_IDS; id++) {
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
       MRC_RMA_SIG_IDS, mr->bytes);
  return -1;
}

static void mrcRmaReleaseSignalId(struct mrcRmaCollComm* cComm, struct mrcRmaMrHandle* mr) {
  if (mr->sigId < 0) return;
  std::lock_guard<std::mutex> lock(cComm->sigMutex);
  cComm->sigMrs[mr->sigId].store(NULL, std::memory_order_release);
  mr->sigId = -1;
}

static ncclResult_t mrcRmaDeregMrSym(void* collComm, void* mhandle);

struct mrcRmaMrExchange {
  uintptr_t va;
  uint32_t rkey;
  int32_t sigId;
};

static ncclResult_t mrcRmaRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                         uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  struct mrcRmaCollComm* cComm = (struct mrcRmaCollComm*)collComm;
  struct mrcRmaMrExchange local, *all = NULL;
  struct mrcRmaMrHandle* mr = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mr, 1), ret, fail);
  mr->sigId = -1;
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

  local.va = (uintptr_t)data;
  local.rkey = mr->mrHandle->mrs[0]->rkey;
  local.sigId = mr->sigId;
  NCCLCHECKGOTO(mrcRmaAllGather(cComm, &local, all, sizeof(local)), ret, fail);
  for (int r = 0; r < cComm->nranks; r++) {
    mr->baseVas[r] = all[r].va;
    mr->rkeys[r] = all[r].rkey;
    if (all[r].sigId != mr->sigId) {
      WARN("NET/MRC/RMA: signal id mismatch for a symmetric registration: rank %d has %d, rank %d has %d",
           cComm->rank, mr->sigId, r, all[r].sigId);
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
    while (sc->ginSeq.completed < sc->ginSeq.posted) {
      struct ibv_wc wc[MRC_RMA_POLL_MAX];
      int nd = 0;
      if (wrap_mrc_poll_cq(sc->devs[0].base.mrcCq, MRC_RMA_POLL_MAX, wc, &nd) != ncclSuccess) return;
      for (int k = 0; k < nd; k++) {
        if (wc[k].status != IBV_WC_SUCCESS) {
          WARN("NET/MRC/RMA: send completion error status=%d to rank %d ctx=%d during teardown", wc[k].status, i,
               context);
          return;
        }
        struct ncclIbRequest* wcReq = sc->base.reqs + wc[k].wr_id;
        if (wcReq->id > sc->ginSeq.completed) sc->ginSeq.completed = wcReq->id;
      }
      clock_gettime(CLOCK_MONOTONIC, &tnow);
      double elapsed = (double)(tnow.tv_sec - t0.tv_sec) + (double)(tnow.tv_nsec - t0.tv_nsec) / 1e9;
      if (elapsed > 10.0) {
        WARN("NET/MRC/RMA: timed out draining %lu outstanding sends to rank %d ctx=%d",
             (unsigned long)(sc->ginSeq.posted - sc->ginSeq.completed), i, context);
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
  }
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
         "it was registered", MRC_RMA_SIG_IDS);
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

static ncclResult_t mrcRmaIGet(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                               uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags,
                               void** request) {
  (void)rmaCtx; (void)context; (void)remoteOff; (void)remoteMhandle; (void)size;
  (void)localOff; (void)localMhandle; (void)rank; (void)optFlags; (void)request;
  WARN("NET/MRC/RMA: get is not supported over MRC (no RDMA read)");
  return ncclInvalidUsage;
}

static ncclResult_t mrcRmaIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  (void)rmaCtx; (void)context; (void)mhandle; (void)rank; (void)request;
  WARN("NET/MRC/RMA: flush is not supported over MRC (no RDMA read)");
  return ncclInvalidUsage;
}

static ncclResult_t mrcRmaTest(void* collComm, void* request, int* done) {
  (void)collComm;
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct mrcRmaCtx* ctx = (struct mrcRmaCtx*)req->rmaProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  // The caller is waiting on this op, so a deferred doorbell must not hold it back.
  NCCLCHECK(mrcRmaPostBatch(ctx));

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ctx->fullSendComm[rank];
  uint64_t* completedSeq = &comm->ginSeq.completed;
  if (req->id > *completedSeq) {
    struct ibv_wc wc[4];
    int wrDone = 0;
    NCCLCHECK(wrap_mrc_poll_cq(comm->devs[0].base.mrcCq, 4, wc, &wrDone));
    for (int i = 0; i < wrDone; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        WARN("NET/MRC/RMA: completion error to rank %d: status=%d opcode=%d len=%u vendor err %u (%s)", rank,
             wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, ncclIbReqTypeStr[req->type]);
        printIbWcStatusHint(wc[i].status);
        return ncclRemoteError;
      }
      struct ncclIbRequest* wcReq = comm->base.reqs + wc[i].wr_id;
      if (wcReq->id > *completedSeq) *completedSeq = wcReq->id;
    }
  }
  if (req->id <= *completedSeq) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
  }
  return ncclSuccess;
}

/* ── Signal delivery ──────────────────────────────────────────────────── */

static ncclResult_t mrcRmaApplySignal(struct mrcRmaCtx* ctx, int rank, struct ibv_wc* wc) {
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
      struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ctx->fullRecvComm[rank];
      if (comm == NULL) continue;
      struct ibv_wc wc[MRC_RMA_POLL_MAX];
      int wrDone = 0;
      NCCLCHECKGOTO(wrap_mrc_poll_cq(comm->devs[0].base.mrcCq, pollCount, wc, &wrDone), ret, exit);
      for (int i = 0; i < wrDone; i++) {
        NCCLCHECKGOTO(mrcRmaApplySignal(ctx, rank, &wc[i]), ret, exit);
        stored = true;
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
