# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build the NCCL NET plugin from the imported include, misc, and transport trees. With RMA=1 (default),
# the MRC RMA plugin in ../rma is linked in as well and exported as ncclRmaPlugin_v17, which NCCL uses as the
# backend of its GIN CPU proxy and host RMA API.

DEBUG ?= 0
MRC_HOME ?= /opt/mellanox/doca
CUDA_HOME ?= /usr/local/cuda
MRC_LIBDIR ?= $(MRC_HOME)/lib/aarch64-linux-gnu
PYTHON ?= python3
RMA ?= 1
RMA_DIR ?= ../rma

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
LDFLAGS += -shared -Wl,-z,defs -Wl,--version-script=$(EXPORT_MAP)
LDFLAGS += -L$(MRC_LIBDIR) -Wl,-rpath,$(MRC_LIBDIR)
LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS += -libverbs -lnv_mrc -lcudart -ldl -lpthread -lrt

OUTPUT ?= ../libnccl-net-mrc.so
OBJDIR ?= ../build/obj
MISC_SRCS := $(filter-out misc/argcheck.cc misc/shmutils.cc misc/strongstream.cc,$(wildcard misc/*.cc))
SRCS := $(MISC_SRCS) $(wildcard os/*.cc) $(wildcard transport/*.cc) $(wildcard transport/*/*.cc)
OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)
ifneq ($(RMA),0)
CPPFLAGS += -I$(RMA_DIR)/include/plugin
RMA_SRCS := $(wildcard $(RMA_DIR)/*.cc)
OBJS += $(RMA_SRCS:$(RMA_DIR)/%.cc=$(OBJDIR)/rma/%.o)
EXPORT_MAP := nccl-net-rma-mrc.map
else
EXPORT_MAP := nccl-net-mrc.map
endif
DEPS := $(OBJS:.o=.d)
GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h
# Records the RMA setting so switching it relinks the library even when no object changed.
CONFIG_STAMP := $(OBJDIR)/rma-config.stamp

all: $(OUTPUT)

$(OUTPUT): $(OBJS) $(EXPORT_MAP) $(CONFIG_STAMP) Makefile
	@printf "Linking    %-35s > %s\n" $(notdir $@) $@
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(OBJDIR)/misc/git_version.o $(OBJDIR)/transport/net_ib/init.o: $(GIT_VERSION_FILE)

$(CONFIG_STAMP): FORCE
	@mkdir -p $(dir $@)
	@echo "RMA=$(RMA)" | cmp -s - $@ || echo "RMA=$(RMA)" > $@

$(GIT_VERSION_FILE): FORCE
	@mkdir -p $(dir $@)
	@$(PYTHON) ./misc/generate_git_version.py $@

$(OBJDIR)/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(OBJDIR)/include -MMD -MP -c $< -o $@

$(OBJDIR)/rma/%.o: $(RMA_DIR)/%.cc
	@printf "Compiling  %-35s > %s\n" $(notdir $<) $@
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(OBJDIR)/include -MMD -MP -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTPUT)

FORCE:

-include $(DEPS)

.PHONY: all clean FORCE
