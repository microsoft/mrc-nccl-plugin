# MRC-Specific Environment Variables

These environment variables control MRC (Multi-Path Reliable Connection) behavior in the NCCL net plugin. They are set with the `NCCL_` prefix (e.g., `NCCL_MRC_TIMEOUT=20`).

## QP Transport Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_MRC_TIMEOUT` | 20 | 0–24 | Local ACK timeout. The actual timeout is 1.024 × 2^value µs. Max value 24 gives ~17.17 s. |
| `NCCL_MRC_RETRY_CNT_LINEAR` | 7 | 0–7 | Linear (fixed-interval) retry limit for lost packets. Currently unsupported by vendor specific mrc.h |
| `NCCL_MRC_RETRY_CNT_EXP` | 25 | 0–25 | Exponential-backoff retry limit. A value of 25 means infinite retries. Currently unsupported by vendor specific mrc.h |
| `NCCL_MRC_QP_HINT_ENABLE` | 1 | 0–1 | Enables creation and attachment of MRC QP hints. Set to 0 to create QPs without hints. |

## Memory Registration Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | 0–2 | Controls PCI relaxed ordering for memory registrations. 0 = disable, 1 = enable, 2 = enable when supported. Registrations requesting strict ordering remain strict. |

## Congestion Control Parameters

| Variable | Default | Range | Description |
|---|---|---|---|
| `NCCL_MRC_CC_INIT_RATE` | 0 | 0+ | Initial congestion-control rate. 0 = no CC hint added. |
| `NCCL_MRC_CC_MIN_RATE` | 0 | 0+ | Minimum congestion-control rate. 0 = no CC hint added. |
| `NCCL_MRC_CC_MAX_RATE` | 0 | 0+ | Maximum congestion-control rate. 0 = no CC hint added. |
