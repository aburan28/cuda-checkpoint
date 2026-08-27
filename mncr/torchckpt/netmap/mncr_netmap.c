/*
 * mncr_netmap - let a migrated process bind to the node it is on now.
 *
 * NCCL discovers the node's address once, at its first initialisation, and
 * keeps it in static memory for the life of the process: the bootstrap
 * interface, and the socket transport's device list. A process restored on a
 * different node still holds the old address, and its first listen after the
 * rebuild fails with EADDRNOTAVAIL - measured, on the second node of a
 * two-node migration, in the NCCL warm-up after an otherwise complete restore.
 *
 * There is no API to reset that cache and no way to reload libnccl from
 * under torch. What there is: NCCL advertises every listener from
 * getsockname() after bind(), not from the cached address. So a bind() that
 * fails with EADDRNOTAVAIL on a unicast IPv4 address is retried on the
 * address this host would use to reach the world - the same choice NCCL
 * itself makes on a fresh node - and every peer learns the new address from
 * the handshake. Nothing else is touched: a bind that succeeds is untouched,
 * wildcard and loopback binds are untouched, IPv6 is untouched.
 *
 * Build:  make -C torchckpt/netmap
 * Use:    LD_PRELOAD=/path/to/libmncr_netmap.so python train.py
 *         MNCR_NETMAP_DEBUG=1 logs each rewrite to stderr.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef int (*bind_fn)(int, const struct sockaddr *, socklen_t);

static bind_fn real_bind(void)
{
    static bind_fn fn;
    if (!fn) {
        fn = (bind_fn)dlsym(RTLD_NEXT, "bind");
    }
    return fn;
}

static int debug_enabled(void)
{
    static int state = -1;
    if (state < 0) {
        const char *v = getenv("MNCR_NETMAP_DEBUG");
        state = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return state;
}

/* The source address the kernel would use for an outbound packet. No packet
 * is sent: a UDP socket "connects" merely to pick a route. Re-evaluated on
 * every call because the answer is exactly what changed. */
static int current_address(struct in_addr *out)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_in probe;
    memset(&probe, 0, sizeof probe);
    probe.sin_family = AF_INET;
    probe.sin_port = htons(1);
    probe.sin_addr.s_addr = inet_addr("10.255.255.255");
    int rc = -1;
    if (connect(fd, (struct sockaddr *)&probe, sizeof probe) == 0) {
        struct sockaddr_in local;
        socklen_t len = sizeof local;
        if (getsockname(fd, (struct sockaddr *)&local, &len) == 0) {
            *out = local.sin_addr;
            rc = 0;
        }
    }
    close(fd);
    return rc;
}

int bind(int fd, const struct sockaddr *addr, socklen_t len)
{
    int rc = real_bind()(fd, addr, len);
    if (rc == 0 || errno != EADDRNOTAVAIL || !addr || addr->sa_family != AF_INET ||
        len < sizeof(struct sockaddr_in)) {
        return rc;
    }
    const struct sockaddr_in *want = (const struct sockaddr_in *)addr;
    uint32_t host = ntohl(want->sin_addr.s_addr);
    if (host == INADDR_ANY || (host >> 24) == 127 || (host >> 28) == 0xE) {
        return rc; /* wildcard, loopback, multicast: not ours to move */
    }

    struct sockaddr_in retry = *want;
    if (current_address(&retry.sin_addr) != 0 ||
        retry.sin_addr.s_addr == want->sin_addr.s_addr) {
        errno = EADDRNOTAVAIL;
        return -1;
    }
    int rc2 = real_bind()(fd, (struct sockaddr *)&retry, sizeof retry);
    if (debug_enabled()) {
        char from[INET_ADDRSTRLEN], to[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &want->sin_addr, from, sizeof from);
        inet_ntop(AF_INET, &retry.sin_addr, to, sizeof to);
        fprintf(stderr, "mncr_netmap: bind %s:%u -> %s:%u %s\n", from,
                ntohs(want->sin_port), to, ntohs(retry.sin_port),
                rc2 == 0 ? "ok" : strerror(errno));
    }
    if (rc2 != 0) {
        errno = EADDRNOTAVAIL;
    }
    return rc2;
}
