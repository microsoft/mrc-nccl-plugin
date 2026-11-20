# MRC RMA plugin (GIN CPU proxy backend)

`ncclRmaPlugin_v17` over MRC. It is linked into `libnccl-net-mrc.so` next to the NET plugin, and stock NCCL
(2.32 or later) uses it as the transport of two NCCL features:

* the **GIN CPU proxy** (`NCCL_GIN_TYPE=2`): device-API kernels post put/signal descriptors to host queues, and
  NCCL's proxy thread turns them into RMA operations;
* the **host RMA API** (`ncclPutSignal`, `ncclSignal`, `ncclWaitSignal`) for peers outside the NVLink domain.

No NCCL changes and no GIN device code are needed: the proxy's device half is NCCL's own. This is the
counterpart of the "Custom GIN CPU Proxy Implementation" box in NCCL's GIN architecture, below the `ncclRma` API.
For GPU-initiated MRC GIN, which needs device code inside NCCL, see the GIN plugin instead.

```
 device-API kernel ──GFD──► NCCL GIN proxy (stock) ──ncclRma v17──► libnccl-net-mrc.so: rma/ ──► net/ net_ib (MRC QPs)
 ncclPutSignal ─────────►  NCCL host RMA proxy (stock) ──────────┘
```

## How NCCL finds it

NCCL loads an RMA backend from, in order: `NCCL_RMA_PLUGIN`, the GIN plugin library, **the NET plugin library**,
then its internal IB and socket backends. Loading `libnccl-net-mrc.so` with `NCCL_NET_PLUGIN` is therefore
enough. With `NCCL_DEBUG=INFO`, every rank logs:

```
RMA/Plugin: Loaded rma plugin MRC (v17)
RMA/Plugin: Assigned plugin MRC to comm
NET/MRC/RMA: MRC RMA plugin initialized (4 devices, GDRCopy on)
GIN: backend MRC using 2 connections on devices [0]mlx5_0 [1]mlx5_1
NET/MRC/RMA: rank 0 created 2 contexts over 16 ranks (16 peers) on mlx5_0
```

`NCCL_RMA_PLUGIN` takes precedence and can select another backend. Build with `make net RMA=0` to leave the RMA
plugin out of the library.

## Design

The plugin reuses the NET plugin's `net_ib` (devices, MRC QPs and CQs, memory registration, request pools), so
both share one device list, one MRC context per NIC and one MR cache per process.

* **Collective comm**: a ring of NET send/recv comms, used for the all-gathers and barriers of setup.
* **Context**: per GIN context (or host RMA context), one MRC send QP and one receive QP per connected peer, self
  included. Only the peers NCCL asks for (`peerArray`) are connected.
* **Put**: an RDMA Write. Work requests are batched per QP and the doorbell is deferred while NCCL hints that more
  requests follow (`ncclRmaOptFlagsAggregateRequests`); only the last WR of a batch is signaled, so send
  completion is tracked per comm with sequence numbers.
* **Put with signal**: MRC has no RDMA atomics, so the signal is the immediate of an RDMA Write-with-Immediate
  that also carries the payload (zero-length for a bare signal). The target's `rmaProgress`, which NCCL calls from
  its proxy threads, reaps the receive completion and adds the value to the signal through a CPU mapping. MRC
  generates the receive completion only after the whole message is placed, so a signal is never visible before
  its data.
* **Receive queues**: each signal consumes a receive WR. Every receive QP keeps 128 buffer-less WRs posted, refilled
  in batches of 32; if `net_ib` pre-posted its own WRs at accept (`NCCL_MRC_PREPOST_RECEIVE_WORK_REQUESTS`, on by
  default), those set the depth.

### Signal buffers

NCCL registers each signal buffer with `NCCL_NET_MR_FLAG_FORCE_SO`. When such a registration is at most
`NCCL_MRC_RMA_SIGNAL_MAP_MAX` bytes, the plugin gives it one of 64 signal ids of its collective comm and maps it for
CPU stores. Registration is collective, so every rank picks the same id; the plugin checks this in the
registration's all-gather.

* GPU buffers are mapped with GDRCopy. On a GPU the CPU reaches over coherent C2C (GB200), the mapping is forced
  onto PCIe (`GDR_PIN_FLAG_FORCE_PCIE`, GDRCopy >= 2.5), the same path as the NIC's payload writes, so the signal
  store is ordered behind the data. Host buffers are written directly.
* Buffers registered with `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET` are only written by the plugin, so it keeps their
  values on the host; others are read, added to and written back.

### Signal immediate

```
 31      26 25                 M+2  M+1   M  M-1          0
[ id (6)   | slot (S bits)         | digit (2) | value (M)  ]     S = ceil(log2(slots)), M = 24 - S
```

The value added is `value << (digit * M)`. An add that needs more than one base-2^M digit is sent as one
immediate per nonzero digit, at most 4, with the payload on the last. Increments always fit in one. A buffer can
have at most 2^23 slots (64 MiB), and an add must be below 2^(4M) (for example 2^24 with 2^18 slots), otherwise the
operation fails with an error.

## Requirements and limitations

* GDRCopy on every node, with FORCE_PCIE support on GB200. Without it, GPU signal buffers cannot be mapped and
  signals to them fail.
* Data Direct is not supported: Data Direct payload writes are not ordered with the CPU's signal stores, and MRC
  has no RDMA Read to flush them. Set `NCCL_IB_DATA_DIRECT=0`; context creation fails on a Data Direct device.
* `get` and `flush` fail (no RDMA Read), and so do signals to arbitrary windows (GIN VA signals). NCCL's GIN
  symmetric kernels use VA signals; keep them off (`NCCL_SYM_GIN_KERNELS_ENABLE=0`) when they would engage.
* At most 64 signal buffers per collective comm can be live at once. Each host RMA context takes three and each
  GIN proxy context one; windows registered with `NCCL_WIN_STRICT_ORDERING` take one each. Host RMA spreads its
  `NCCL_NUM_RMA_CTX` user contexts and `NCCL_NUM_RMA_INT_CTX` (default 4) internal ones over one collective comm
  per NIC. Running out is reported when the buffer is registered.
* `NCCL_IB_HCA` needs the rail and plane IDs on fabrics with disjoint planes, as for the NET plugin.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NCCL_GIN_IB_TC` | -1 | Traffic class of the RMA QPs; -1 uses `NCCL_IB_TC`. |
| `NCCL_RMA_IB_WR_BATCHSIZE` | 64 | Maximum work requests per doorbell (4 to 64). |
| `NCCL_MRC_RMA_POLL_COUNT` | 16 | Receive completions reaped per peer per progress call (1 to 64). |
| `NCCL_MRC_RMA_SIGNAL_MAP_MAX` | 67108864 | Largest `FORCE_SO` registration, in bytes, that can be a signal target. |

## Running

```bash
mpirun ... \
  -x NCCL_NET_PLUGIN=/path/to/libnccl-net-mrc.so -x NCCL_NET=MRC \
  -x NCCL_GIN_TYPE=2 -x NCCL_IB_DATA_DIRECT=0 \
  -x NCCL_IB_HCA==mlx5_1:1:0:1,mlx5_0:1:0:0,mlx5_3:1:1:1,mlx5_2:1:1:0 \
  <binary>
```

Device-API applications need no special build flags: they use NCCL's proxy device code.
