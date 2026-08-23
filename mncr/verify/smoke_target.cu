/*
 * smoke_target - a minimal CUDA process for bring-up testing.
 *
 * Deliberately not PyTorch: when you are validating that the driver and CRIU
 * behave on this node, an allocator and a communications library are variables
 * you do not want. It allocates device memory, fills it with a known pattern,
 * and answers checksum requests through a file so the harness can prove the
 * data survived a checkpoint.
 *
 * File protocol, chosen so the process holds nothing across a dump but its own
 * memory - no sockets, no pipes, nothing whose peer lives outside the tree:
 *
 *   <state>          written by us:  "ready <pid>" then "sum <value>"
 *   <state>.cmd      read by us:     "verify" | "exit"
 *
 * Build:  nvcc -O2 smoke_target.cu -o smoke_target
 * Run:    ./smoke_target /tmp/smoke.state [elements]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define CHECK(x)                                                               \
    do {                                                                       \
        cudaError_t rc = (x);                                                  \
        if (rc != cudaSuccess) {                                               \
            fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(rc));    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void fill(unsigned int *buf, size_t n, unsigned int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buf[i] = seed + (unsigned int)i * 2654435761u;
    }
}

static void write_state(const char *path, const char *text)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *fh = fopen(tmp, "w");
    if (!fh) {
        return;
    }
    fputs(text, fh);
    fclose(fh);
    rename(tmp, path);
}

static int read_command(const char *path, char *out, size_t len)
{
    char cmd_path[4096];
    snprintf(cmd_path, sizeof cmd_path, "%s.cmd", path);
    FILE *fh = fopen(cmd_path, "r");
    if (!fh) {
        return 0;
    }
    if (!fgets(out, (int)len, fh)) {
        out[0] = '\0';
    }
    fclose(fh);
    unlink(cmd_path);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <state-file> [elements]\n", argv[0]);
        return 2;
    }
    const char *state = argv[1];
    size_t elements = (argc > 2) ? (size_t)atol(argv[2]) : (16u << 20);
    const unsigned int seed = 0x5A5A0001u;

    unsigned int *device = NULL;
    CHECK(cudaMalloc(&device, elements * sizeof *device));

    int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    fill<<<blocks, threads>>>(device, elements, seed);
    CHECK(cudaDeviceSynchronize());

    char line[128];
    snprintf(line, sizeof line, "ready %d %zu\n", (int)getpid(), elements);
    write_state(state, line);

    unsigned int *host = (unsigned int *)malloc(elements * sizeof *host);
    if (!host) {
        return 1;
    }

    for (;;) {
        char cmd[64] = {0};
        if (read_command(state, cmd, sizeof cmd)) {
            if (strncmp(cmd, "exit", 4) == 0) {
                break;
            }
            if (strncmp(cmd, "verify", 6) == 0) {
                CHECK(cudaMemcpy(host, device, elements * sizeof *host,
                                 cudaMemcpyDeviceToHost));
                unsigned long long sum = 0;
                int bad = 0;
                for (size_t i = 0; i < elements; i++) {
                    unsigned int want = seed + (unsigned int)i * 2654435761u;
                    if (host[i] != want) {
                        bad++;
                    }
                    sum += host[i];
                }
                snprintf(line, sizeof line, "sum %llu bad %d pid %d\n", sum, bad,
                         (int)getpid());
                write_state(state, line);
            }
        }
        usleep(50 * 1000);
    }

    free(host);
    cudaFree(device);
    return 0;
}
