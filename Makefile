TARGET ?= libnccl-net-mrc.so

all:
	$(MAKE) -C src OUTPUT=../$(TARGET)

clean:
	$(MAKE) -C src OUTPUT=../$(TARGET) clean

test: all
	$(MAKE) -C tests PLUGIN=$(abspath $(TARGET)) test

.PHONY: all clean test
