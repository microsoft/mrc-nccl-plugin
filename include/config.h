// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Minimal config header for Makefile-only builds.
//
// Upstream versions of this plugin use an autotools-generated `config.h`
// which defines `HAVE_*` / `HAVE_DECL_*` feature macros based on the installed
// rdma-core headers.
//
// In this workspace we build with a plain Makefile, so we provide conservative
// defaults suitable for modern rdma-core environments (e.g. the target host
// 10.0.5.247 where these APIs exist in the headers). Override any of these at
// build time via `CPPFLAGS+=-DMACRO=0/1` if needed.

#ifndef NCCL_RDMA_PLUGIN_CONFIG_H_
#define NCCL_RDMA_PLUGIN_CONFIG_H_

// infiniband/mlx5dv.h availability
#ifndef HAVE_INFINIBAND_MLX5DV_H
#define HAVE_INFINIBAND_MLX5DV_H 1
#endif

// DMA-BUF registration APIs
#ifndef HAVE_DECL_IBV_REG_DMABUF_MR
#define HAVE_DECL_IBV_REG_DMABUF_MR 1
#endif

#ifndef HAVE_DECL_MLX5DV_REG_DMABUF_MR
#define HAVE_DECL_MLX5DV_REG_DMABUF_MR 1
#endif

#ifndef HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH
#define HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH 1
#endif

#endif
