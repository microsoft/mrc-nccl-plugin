# mrc-nccl-plugin

External NCCL plugins for MRC (Multi-Path Reliable Connection).

| Plugin | Library | Source | Make target |
|--------|---------|--------|-------------|
| NET | `libnccl-net-mrc.so` | [net/](net/) | `make net` |
| RMA (GIN CPU proxy backend) | `libnccl-net-mrc.so` | [rma/](rma/) | `make net` |
| Tuner | `libnccl-tuner-mrc.so` | [tuner/](tuner/) | `make tuner` |

`make` (or `make all`) builds every plugin. To build only some of them, use their make targets (for example, `make tuner`) or set `PLUGINS` to a space-separated list (for example, `make PLUGINS=tuner`). `make clean` removes the build output for all plugins.

## NET plugin

The NET plugin uses MRC as the transport. This rebased C++ branch builds the NCCL NET v12 ABI and reports the network name `MRC`. It requires a NCCL runtime that supports v12; the legacy C sources are not part of this build.


### Requirements

* MRC
* CUDA
* OFED
* NCCL with NET v12 support (runtime)
* A C++17 compiler, Make, and Python 3 (build)

### Build

```bash
make net MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda
```
Debug build
```bash
make net MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda DEBUG=1
```

To supply `nccl.h` from a custom NCCL installation, add `NCCL_HOME=/path/to/nccl/build` to the make command. `nccl.h` will be included from `$(NCCL_HOME)/include`.

`libnccl-net-mrc.so` will be generated after building. `mrc` is the suffix of the plugin library.

### Run (NCCL)

NCCL loads external plugins via the `NCCL_NET_PLUGIN` environment variable. If `NCCL_NET_PLUGIN` is set as `libnccl-net-mrc.so` or just as the suffix `mrc`, add the folder containing the plugin library to your `LD_LIBRARY_PATH`. Instead, `NCCL_NET_PLUGIN` can be set to the absolute path, for e.g., `/path/to/libnccl-net-mrc.so`.

With `NCCL_DEBUG=INFO`, verify `Loaded net plugin MRC (v12)`, `Using network MRC`. For the version of mrc-nccl-plugin being used, look for `Initializing MRC plugin version` string. The plugin version and the git version will follow it. A `NET/Plugin` load error followed by `Using network IB` means NCCL fell back to its built-in transport; subsequent IB CQE errors do not establish a failure in the MRC data path. Setting `NCCL_NET=MRC` (and forwarding it to every rank) makes such fallback fail explicitly instead.

## RMA plugin

The RMA plugin (`ncclRmaPlugin_v17`) is built into `libnccl-net-mrc.so` with the NET plugin, and NCCL loads it from
the NET plugin library. Stock NCCL (2.32 or later) then runs its GIN CPU proxy (`NCCL_GIN_TYPE=2`) and its host RMA
API (`ncclPutSignal`, `ncclWaitSignal`) over MRC, with signals carried as RDMA Write-with-Immediate. It needs
GDRCopy and `NCCL_IB_DATA_DIRECT=0`. Build with `make net RMA=0` to leave it out. See [rma/README.md](rma/README.md).

## Tuner plugin

The tuner plugin selects NCCL algorithms and protocols from user-supplied CSV rules in `nccl_tuner.conf`, and falls back to a GB200F cost model where a validated profile applies (currently `rail_nop2p`; MNNVL jobs use NCCL's defaults). It needs no MRC, CUDA, or NCCL to build:

```bash
make tuner
```

This creates `libnccl-tuner-mrc.so`. To load it, set `NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner-mrc.so`. See [tuner/README.md](tuner/README.md) for configuration, environment variables, and the test drivers (`make test-tuner`).
