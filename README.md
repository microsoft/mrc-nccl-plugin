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
