NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
HOST_WARNINGS ?= -Wall,-Wextra

.PHONY: all clean

all: gds-transport-benchmark

gds-transport-benchmark: gds-transport-benchmark.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Xcompiler=$(HOST_WARNINGS) $< -o $@ $(LDFLAGS) -lcufile -lcuda $(LDLIBS)

clean:
	$(RM) gds-transport-benchmark
