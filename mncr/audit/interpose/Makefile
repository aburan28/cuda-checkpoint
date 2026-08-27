# cuda_audit.so - P0 allocation auditor.
#
# Builds without the CUDA toolkit. Set USE_CUDA_H=1 to compile against the real
# headers instead of the minimal local declarations.
#
# `make test` runs the interposer against a fake libcuda, which is the only way
# to exercise it without a GPU. It proves the part that is easy to get wrong:
# the cuGetProcAddress redirect, without which an audit of a real workload comes
# back clean because NCCL never touches the PLT entry.

CC      ?= cc
CFLAGS  ?= -O2 -g -fPIC -Wall -Wextra -Wno-unused-parameter
LDLIBS  ?= -ldl -lpthread

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  SO     := dylib
  SHARED := -dynamiclib
  LDLIBS := -lpthread
  # macOS has no working equivalent of LD_PRELOAD for this: two-level
  # namespaces bind each call to its defining library at link time, and
  # DYLD_FORCE_FLAT_NAMESPACE is inert on current releases. So the local test
  # links the probe against the interposer directly and points it at the fake
  # driver. That covers all of the interposer's logic - recording, severity,
  # and the cuGetProcAddress redirect - and leaves only the preload mechanism
  # itself to Linux, where it is the production path and CI exercises it.
  PROBE_LIBS = $(CURDIR)/cuda_audit.$(SO) $(CURDIR)/test/libfakecuda.$(SO)
  RUN_ENV    = MNCR_AUDIT_REAL_LIB=$(CURDIR)/test/libfakecuda.$(SO)
  MODE        := direct-link
  EXPECT_ARGS :=
else
  SO     := so
  SHARED := -shared
  # The production mechanism, unmodified.
  PROBE_LIBS = -L$(CURDIR)/test -lfakecuda
  RUN_ENV    = LD_PRELOAD=$(CURDIR)/cuda_audit.$(SO)
  MODE        := LD_PRELOAD
  EXPECT_ARGS := --expect-dlsym-hook
endif

ifeq ($(USE_CUDA_H),1)
CFLAGS += -DMNCR_USE_CUDA_H -I$(CUDA_HOME)/include
endif

all: cuda_audit.$(SO)

cuda_audit.$(SO): cuda_audit.c
	$(CC) $(CFLAGS) $(SHARED) -o $@ $< $(LDLIBS)

# Parse-only check, for hosts without a working toolchain.
check: cuda_audit.c
	$(CC) $(CFLAGS) -fsyntax-only $<

test/libfakecuda.$(SO): test/fake_libcuda.c
	$(CC) $(CFLAGS) $(SHARED) -o $@ $<

test/probe: test/probe.c test/libfakecuda.$(SO) cuda_audit.$(SO)
	$(CC) $(CFLAGS) -DLIBSUFFIX='".$(SO)"' -o $@ test/probe.c $(PROBE_LIBS) \
	  -ldl -Wl,-rpath,$(CURDIR) -Wl,-rpath,$(CURDIR)/test

.PHONY: test
test: cuda_audit.$(SO) test/probe
	@echo "interposer test mode: $(MODE)"
	@rm -f /tmp/mncr-audit-test*
	@$(RUN_ENV) MNCR_AUDIT_OUT=/tmp/mncr-audit-test ./test/probe > /dev/null
	@python3 test/assert_audit.py /tmp/mncr-audit-test $(EXPECT_ARGS)
	@rm -f /tmp/mncr-audit-test*

clean:
	rm -f cuda_audit.so cuda_audit.dylib test/probe test/libfakecuda.so \
	      test/libfakecuda.dylib
	rm -rf test/*.dSYM

.PHONY: all check clean
