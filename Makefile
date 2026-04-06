# Minimal build for an external NCCL NET plugin.
#

DEBUG ?= 0
MRC_HOME ?= /opt/mellanox/doca
CUDA_HOME ?= /usr/local/cuda
MRC_LIBDIR ?= $(MRC_HOME)/lib/aarch64-linux-gnu
NCCL_HOME ?= /path/to/nccl/build

CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -std=gnu11
ifneq ($(DEBUG),0)
CFLAGS += -g
endif

CPPFLAGS += -Iinclude -I$(NCCL_HOME)/include -I$(MRC_HOME)/include -I$(CUDA_HOME)/include
LDFLAGS  += -shared
LDFLAGS  += -L$(MRC_LIBDIR) -Wl,-rpath,$(MRC_LIBDIR)
LDLIBS   += -libverbs -lnv_mrc -ldl -lpthread 

# Only needed if something in this plugin ends up referencing CUDA runtime
# symbols (most builds won't). Keeping it as an opt-in knob.
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64
CUDA_LDLIBS ?= -lcudart

TARGET = libnccl-net-mrc.so
SRCS   = \
	src/net_mrc_plugin.c \
	src/p2p_plugin.c \
	src/param.c \
	src/utils.c \
	src/socket.c \
	src/ibvwrap.c
OBJS   = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
