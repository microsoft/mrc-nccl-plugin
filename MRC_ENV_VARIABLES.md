# MRC-Specific Environment Variables

These environment variables control MRC (Multi-Path Reliable Connection) behavior in the NCCL net plugin. They are set with the `NCCL_` prefix (e.g., `NCCL_MRC_TIMEOUT=20`).

## QP Transport Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_IB_MIN_RNR_TIMER` | 12 | 0–31 | Minimum delay before retrying after an RNR NAK. The default value of 12 corresponds to 0.64 ms. |
| `NCCL_MRC_TIMEOUT` | 20 | 0–24 | Local ACK timeout. The actual timeout is 1.024 × 2^value µs. Max value 24 gives ~17.17 s. |
| `NCCL_MRC_RETRY_CNT_LINEAR` | 7 | 0–7 | Linear (fixed-interval) retry limit for lost packets. Currently unsupported by vendor specific mrc.h |
| `NCCL_MRC_RETRY_CNT_EXP` | 25 | 0–25 | Exponential-backoff retry limit. A value of 25 means infinite retries. Currently unsupported by vendor specific mrc.h |
| `NCCL_MRC_QP_HINT_ENABLE` | 0 | 0–1 | Enables a hint for each MRC data/CTS QP on both send and receive connections. Set to 0 to skip all QP and CC hints. Ordinary verbs GPU-flush QPs never receive MRC hints. |
| `NCCL_MRC_MULTI_RECV_ENABLE` | 1 | 0–1 | Enables grouped receives of up to eight tags. Enabled by default. Set `NCCL_MRC_MULTI_RECV_ENABLE=0` to disable grouped receives. |
| `NCCL_MRC_PREPOST_RECEIVE_WORK_REQUESTS` | 1 | 0 or 1 | Set to 0 to disable preposting. Leave as 1 to enable preposting. |

## Memory Registration Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | 0–2 | Controls PCI relaxed ordering for memory registrations. 0 = disable, 1 = enable, 2 = enable when supported. Registrations requesting strict ordering remain strict. |

## Congestion Control Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_MRC_CC_INIT_RATE` | 0 | 0–1048576 | Initial congestion-control rate, passed unchanged to each QP. |
| `NCCL_MRC_CC_MIN_RATE` | 0 | 0-1048576 | Connection-wide minimum, divided (integer division) by the total negotiated QP count across all merged devices. |
| `NCCL_MRC_CC_MAX_RATE` | 0 | 0–1048576 | Maximum congestion-control rate, passed unchanged to each QP. |

The rebased plugin preserves the legacy version-1 CC payload: if any effective rate is nonzero, all three are packed into the hint's vendor configuration. If all are zero, the vendor configuration remains zero and provider CC defaults apply, but topology hints are still created when `NCCL_MRC_QP_HINT_ENABLE=1`.

Each hint advertises `num_qps_per_peer` equal to the connection's negotiated QP count and `num_send_peers=1`. It is attached during INIT and destroyed only after its QP. As in the legacy plugin, hint creation/attachment failures fail the connection; the requested configuration is not silently discarded. `NCCL_DEBUG=INFO` with the `NET` subsystem logs the effective rates. The rate units are provider-specific, not bits per second (the NVIDIA provider uses 1048576 for line rate).

## Rebased Plugin Diagnostics

- The initialization banner uses NCCL `INFO` logging once on global rank zero, identified from `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `RANK`, or `SLURM_PROCID` (in that order). It respects `NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS`. When no global rank is known, the optional banner is omitted rather than repeated on every process.

## Speed scaling

Since MRC-enabled VF is created from first PF, it inherits the speed of just the first PF. `NCCL_MRC_NUM_PLANES` scales the effective speed by the number of planes. Its default is 8.
