# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build the NCCL NET plugin from the imported include, misc, and transport trees.

DEBUG ?= 0
MRC_HOME ?= /opt/mellanox/doca
CUDA_HOME ?= /usr/local/cuda
MRC_LIBDIR ?= $(MRC_HOME)/lib/aarch64-linux-gnu
PYTHON ?= python3

CXX ?= g++
ifneq ($(NCCL_HOME),)
CPPFLAGS += -I$(NCCL_HOME)/include
endif
CPPFLAGS += -I. -Iinclude -Iinclude/plugin -I$(MRC_HOME)/include
CPPFLAGS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
CPPFLAGS += -DNCCL_OS_LINUX -DNCCL_BUILD_RDMA_CORE -DNCCL_COMM_H_ -DncclNetIb=ncclNetPlugin_v12
CXXFLAGS ?= -O3 -g -fPIC -Wall -Wno-unused-function -Wno-sign-compare -Wvla -std=c++17
ifneq ($(DEBUG),0)
CXXFLAGS := $(filter-out -O3,$(CXXFLAGS)) -O0 -ggdb3
endif

# An external plugin cannot resolve private symbols from libnccl. Fail the
# build instead of letting dlopen fail and NCCL silently fall back to IB.
LDFLAGS += -shared -Wl,-z,defs -Wl,--version-script=nccl-net-mrc.map
LDFLAGS += -L$(MRC_LIBDIR) -Wl,-rpath,$(MRC_LIBDIR)
LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS += -libverbs -lnv_mrc -lcudart -ldl -lpthread -lrt

OUTPUT ?= ../libnccl-net-mrc.so
OBJDIR ?= ../build/obj
MISC_SRCS := $(filter-out misc/argcheck.cc misc/shmutils.cc misc/strongstream.cc,$(wildcard misc/*.cc))
SRCS := $(MISC_SRCS) $(wildcard os/*.cc) $(wildcard transport/*.cc) $(wildcard transport/*/*.cc)
OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)
DEPS := $(OBJS:.o=.d)
GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h

all: $(OUTPUT)

$(OUTPUT): $(OBJS) nccl-net-mrc.map Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(OBJDIR)/misc/git_version.o $(OBJDIR)/transport/net_ib/init.o: $(GIT_VERSION_FILE)

$(GIT_VERSION_FILE): FORCE
	@mkdir -p $(dir $@)
	@$(PYTHON) ./misc/generate_git_version.py $@

$(OBJDIR)/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(OBJDIR)/include -MMD -MP -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTPUT)

FORCE:

-include $(DEPS)

.PHONY: all clean FORCE
