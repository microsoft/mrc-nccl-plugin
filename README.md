# mrc-nccl-plugin

An external NCCL network plugin using MRC (Multi-Path Reliable Connection) as transport. It supports NCCL NET plugin ABI versions v6 through v11.


## Requirements

* MRC
* CUDA
* OFED
* NCCL

## Build

```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl/build
```
Debug build
```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl/build DEBUG=1
```

libnccl-net-mrc.so will be generated after building, with `mrc` as the suffix of the plugin library.

## Run (NCCL)

NCCL loads external plugins via the `NCCL_NET_PLUGIN` environment variable. It can be set to either
a suffix string or to a library name.

Example:

```bash
export NCCL_NET_PLUGIN=$PWD/libnccl-net-mrc.so
```
or
```bash
export NCCL_NET_PLUGIN=mrc
```

## Tests

The [`tests/`](tests/) directory contains MPI launcher scripts that run the upstream
[nccl-tests](https://github.com/NVIDIA/nccl-tests) benchmarks against the MRC NCCL plugin.

| Script | Purpose |
| --- | --- |
| [`tests/run-nccl-mrc-only.sh`](tests/run-nccl-mrc-only.sh) | Run a benchmark with the MRC plugin only (SHM/P2P/MNNVL disabled). Default sweep: `4K..4G`. |
| [`tests/run-nccl-mrc-mnnvl.sh`](tests/run-nccl-mrc-mnnvl.sh) | Run a benchmark with the MRC plugin and MNNVL enabled (SHM/P2P enabled, `NCCL_MNNVL_ENABLE=1`). Default sweep: `1M..16G`. |

### Usage

Both scripts share the same interface:

```bash
./tests/run-nccl-mrc-only.sh  <NUM_NODES> <BENCH>
./tests/run-nccl-mrc-mnnvl.sh <NUM_NODES> <BENCH>
```

* `NUM_NODES` — number of nodes listed in the MPI hostfile.
* `BENCH` — nccl-tests benchmark name (e.g. `sendrecv`, `all_reduce`, `all_gather`,
  `reduce_scatter`, `broadcast`, `reduce`, `alltoall`). The script runs
  `${NCCL_TESTS_DIR}/${BENCH}_perf`.

Example:

```bash
HOSTFILE=~/hostfile ./tests/run-nccl-mrc-only.sh 2 sendrecv
```

### Environment overrides


| Variable | Default | Description |
| --- | --- | --- |
| `PPN` | `4` | Processes per node. |
| `HOSTFILE` | `./hostfile` | Path to the MPI hostfile. |
| `MPI_MODULE` | `mpi/hpcx` | MPI module to load via `module load`. Set to empty to skip. |
| `NCCL_MRC_PLUGIN_PATH` | `/opt/microsoft/mrc/mrc-nccl-plugin/` | Directory containing `libnccl-net-mrc.so`. Prepended to `LD_LIBRARY_PATH`. |
| `NCCL_TESTS_DIR` | `/opt/nccl-tests/build` | Directory containing built nccl-tests binaries. |
| `MPI_NETDEV` | `enP22p1s0f1` | Frontend/MANA NIC for MPI/UCX/NCCL bootstrap. |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPU ordering. |
| `NCCL_IB_HCA` | `=mlx5_1,mlx5_0,mlx5_3,mlx5_2` | IB HCA list. |
| `GID_INDEX` | `3` | RoCE GID index (`NCCL_IB_GID_INDEX`). |
| `WARMUP_ITER` | `50` | Warmup iterations (`-w`). |
| `INNER_ITER` | `50` | Inner iterations (`-n`). |
| `OUTER_ITER` | `1` | Outer iterations (`-N`). |
| `BEGIN_MSG_SIZE` | `4K` (only) / `1M` (mnnvl) | Starting message size (`-b`). |
| `END_MSG_SIZE` | `4G` (only) / `16G` (mnnvl) | Ending message size (`-e`); also the held size in long-running mode. |
| `LONG_RUN` | `0` | When `1`, run indefinitely at `END_MSG_SIZE` (`-N 0`). |
| `DEBUG` | `0` | When `1`, set `NCCL_DEBUG=TRACE` with `NCCL_DEBUG_SUBSYS=ALL` (otherwise `WARN`). |
| `ADDNL_MPI_ARGS` | *(empty)* | Extra arguments appended to `mpirun`. |
| `ADDNL_NCCL_ENV` | *(empty)* | Extra `-x VAR=value` entries appended to the NCCL env. |

`NCCL_TESTS_SPLIT_MASK` is derived automatically from `PPN` (`0x0` / `0x1` / `0x3` for
`PPN`=1/2/4) when running `tests/run-nccl-mrc-only.sh`

### Examples

Run `all_reduce` on 8 nodes with 2 processes per node and a custom hostfile:

```bash
PPN=2 HOSTFILE=./my-hosts ./tests/run-nccl-mrc-only.sh 8 all_reduce
```

Long-running soak at 4G with debug tracing:

```bash
LONG_RUN=1 DEBUG=1 ./tests/run-nccl-mrc-only.sh 4 sendrecv
```

Override the plugin and nccl-tests locations:

```bash
NCCL_MRC_PLUGIN_PATH=$PWD \
NCCL_TESTS_DIR=$HOME/nccl-tests/build \
./tests/run-nccl-mrc-only.sh 2 sendrecv
```

