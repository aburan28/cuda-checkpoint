# cuda_audit.so - P0 allocation auditor.
#
# Builds without the CUDA toolkit. Set USE_CUDA_H=1 to compile against the real
# headers instead of the minimal local declarations.

CC      ?= cc
CFLAGS  ?= -O2 -g -fPIC -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -shared
LDLIBS  ?= -ldl -lpthread

ifeq ($(USE_CUDA_H),1)
CFLAGS += -DMNCR_USE_CUDA_H -I$(CUDA_HOME)/include
endif

all: cuda_audit.so

cuda_audit.so: cuda_audit.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

# Parse-only check, for hosts without a Linux toolchain.
check: cuda_audit.c
	$(CC) $(CFLAGS) -fsyntax-only $<

clean:
	rm -f cuda_audit.so

.PHONY: all check clean
