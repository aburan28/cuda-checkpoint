# libmncr_netmap.so - see mncr_netmap.c. Linux only; a rank preloads it.
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
LIB      = libmncr_netmap.so

.PHONY: all test clean
all: $(LIB)

# A fixed mode, deliberately: a rank maps this file, and criu refuses to
# restore onto a node where the same path has a different mode - measured,
# when one node's copy was built under umask 002 and the other's under 022.
$(LIB): mncr_netmap.c
	$(CC) $(CFLAGS) -shared -o $@ $< -ldl
	chmod 0755 $@

# Binds to an address this host does not have, which must be rewritten, and
# to loopback, which must not be.
test: $(LIB)
	python3 test_netmap.py ./$(LIB)

clean:
	rm -f $(LIB)
