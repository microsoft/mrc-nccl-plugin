# mrc-nccl-plugin

An external NCCL network plugin using MRC (Multi-Path Reliable Connection) as transport. This rebased C++ branch builds the NCCL NET v12 ABI and reports the network name `MRC`. It requires a NCCL runtime that supports v12; the legacy C sources are not part of this build.


## Requirements

* MRC
* CUDA
* OFED
* NCCL with NET v12 support (runtime)
* A C++17 compiler, Make, and Python 3 (build)

## Build

```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda
```
Debug build
```bash
make MRC_HOME=/path/to/mrc CUDA_HOME=/path/to/cuda DEBUG=1
```

To supply `nccl.h` from a custom NCCL installation, add `NCCL_HOME=/path/to/nccl/build` to the make command. `nccl.h` will be included from `$(NCCL_HOME)/include`.

`libnccl-net-mrc.so` will be generated after building. `mrc` is the suffix of the plugin library.

## Run (NCCL)

NCCL loads external plugins via the `NCCL_NET_PLUGIN` environment variable. If `NCCL_NET_PLUGIN` is set as `libnccl-net-mrc.so` or just as the suffix `mrc`, add the folder containing the plugin library to your `LD_LIBRARY_PATH`. Instead, `NCCL_NET_PLUGIN` can be set to the absolute path, for e.g., `/path/to/libnccl-net-mrc.so`.

With `NCCL_DEBUG=INFO`, verify `Loaded net plugin MRC (v12)`, `Using network MRC`. For the version of mrc-nccl-plugin being used, look for `Initializing MRC plugin version` string. The plugin version and the git version will follow it. A `NET/Plugin` load error followed by `Using network IB` means NCCL fell back to its built-in transport; subsequent IB CQE errors do not establish a failure in the MRC data path. Setting `NCCL_NET=MRC` (and forwarding it to every rank) makes such fallback fail explicitly instead. 
