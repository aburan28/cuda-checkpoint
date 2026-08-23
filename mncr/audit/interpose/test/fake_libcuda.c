/*
 * A stand-in for libcuda, so the interposer can be tested without a GPU.
 *
 * Exports the entry points cuda_audit.so hooks, each returning success, plus a
 * cuGetProcAddress that resolves them the way the real driver does. That last
 * one is the point of the whole exercise: since CUDA 12 the runtime and NCCL
 * fetch driver entry points through cuGetProcAddress, which hands back a
 * pointer from inside libcuda and never touches the interposer's PLT entry. If
 * the redirect is broken, an audit of a real workload comes back clean and
 * nobody finds out until a checkpoint fails.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;

int fake_call_count;

#define FAKE(name, ...)                                                        \
    CUresult name(__VA_ARGS__)                                                 \
    {                                                                          \
        fake_call_count++;                                                     \
        return 0;                                                              \
    }

FAKE(cuMemCreate, void *h, size_t size, const void *prop, unsigned long long f)
FAKE(cuMemMap, unsigned long long p, size_t s, size_t o, void *h, unsigned long long f)
FAKE(cuMemExportToShareableHandle, void *out, void *h, int type, unsigned long long f)
FAKE(cuMemImportFromShareableHandle, void *h, void *osh, int type)
FAKE(cuMemAllocManaged, unsigned long long *p, size_t s, unsigned int f)
FAKE(cuMulticastCreate, void *h, const void *prop)
FAKE(cuMulticastBindMem, void *mc, size_t mo, void *m, size_t o, size_t s,
     unsigned long long f)
FAKE(cuIpcGetMemHandle, void *h, unsigned long long p)
FAKE(cuIpcOpenMemHandle, unsigned long long *p, void *h, unsigned int f)

/* The real driver hands back its own internal pointers. We hand back ours; the
 * interposer is expected to substitute its wrappers by name. */
struct entry {
    const char *name;
    void *fn;
};

static const struct entry g_entries[] = {
    {"cuMemCreate", (void *)cuMemCreate},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemExportToShareableHandle", (void *)cuMemExportToShareableHandle},
    {"cuMemImportFromShareableHandle", (void *)cuMemImportFromShareableHandle},
    {"cuMemAllocManaged", (void *)cuMemAllocManaged},
    {"cuMulticastCreate", (void *)cuMulticastCreate},
    {"cuMulticastBindMem", (void *)cuMulticastBindMem},
    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle},
};

CUresult cuGetProcAddress(const char *symbol, void **pfn, int version, uint64_t flags)
{
    (void)version;
    (void)flags;
    for (size_t i = 0; i < sizeof g_entries / sizeof g_entries[0]; i++) {
        if (strcmp(g_entries[i].name, symbol) == 0) {
            *pfn = g_entries[i].fn;
            return 0;
        }
    }
    return 500; /* CUDA_ERROR_NOT_FOUND */
}

CUresult cuDriverGetVersion(int *version)
{
    *version = 12080;
    return 0;
}
