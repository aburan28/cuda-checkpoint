/*
 * cuda_audit.so - LD_PRELOAD interposer that reports allocations the CUDA
 * checkpoint path cannot handle.
 *
 * P0 of the build plan. The point is to answer, for a real training or serving
 * image, exactly which unsupported resources the process holds - before anyone
 * writes teardown code against a guess.
 *
 * What it watches, and why each one matters:
 *
 *   cuMemCreate / cuMemMap              VMM allocations. PyTorch's expandable
 *                                       segments and NCCL's default allocator
 *                                       both land here.
 *   cuMemExportToShareableHandle        The documented hard limitation. POSIX-fd
 *                                       and FABRIC handle types alike.
 *   cuMemImportFromShareableHandle      The receiving side of the same.
 *   cuMemAllocManaged                   UVM, also unsupported.
 *   cuMulticastCreate / cuMulticastBind NVLS multicast objects. Restore errors
 *                                       in libcuda if these are still live.
 *   cuIpcGetMemHandle                   Legacy IPC. Supported from driver 610,
 *                                       but only within a launched job, so it
 *                                       is worth knowing about.
 *
 * Interposing on the symbols alone is not enough. Since CUDA 12 the runtime and
 * NCCL resolve driver entry points through cuGetProcAddress, which returns a
 * pointer straight out of libcuda and never touches our PLT entry. So we hook
 * cuGetProcAddress as well and hand back our wrappers by name. Missing this is
 * the difference between "the audit came back clean" and "the audit saw
 * nothing".
 *
 * Build:  make
 * Use:    MNCR_AUDIT_OUT=/tmp/audit.jsonl LD_PRELOAD=./cuda_audit.so ./your_app
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__GLIBC__) || defined(__APPLE__)
#include <execinfo.h>
#define HAVE_BACKTRACE 1
#endif

/* Minimal CUDA surface so this builds without the toolkit installed. The real
 * headers are used when they are available. */
#ifdef MNCR_USE_CUDA_H
#include <cuda.h>
#else
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUmemGenericAllocationHandle;
typedef void *CUmemAllocationProp;
typedef void *CUmemAllocationHandleType_ptr;
typedef struct CUipcMemHandle_st { char reserved[64]; } CUipcMemHandle;
typedef void *CUmulticastObjectProp;
#endif

#define MAX_FRAMES 24

/* ------------------------------------------------------------------ output */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_out;
static int g_backtrace;
static int g_ready;

static struct {
    const char *name;
    unsigned long count;
} g_counts[] = {
    {"cuMemCreate", 0},
    {"cuMemMap", 0},
    {"cuMemExportToShareableHandle", 0},
    {"cuMemImportFromShareableHandle", 0},
    {"cuMemAllocManaged", 0},
    {"cuMulticastCreate", 0},
    {"cuMulticastBindMem", 0},
    {"cuMulticastBindAddr", 0},
    {"cuIpcGetMemHandle", 0},
    {"cuIpcOpenMemHandle", 0},
};
static const int g_ncounts = (int)(sizeof g_counts / sizeof g_counts[0]);

static void bump(const char *name)
{
    for (int i = 0; i < g_ncounts; i++) {
        if (strcmp(g_counts[i].name, name) == 0) {
            g_counts[i].count++;
            return;
        }
    }
}

static void emit_summary(void)
{
    if (!g_out)
        return;
    pthread_mutex_lock(&g_lock);
    fprintf(g_out, "{\"event\":\"summary\",\"pid\":%d,\"counts\":{", (int)getpid());
    int first = 1;
    for (int i = 0; i < g_ncounts; i++) {
        if (g_counts[i].count == 0)
            continue;
        fprintf(g_out, "%s\"%s\":%lu", first ? "" : ",", g_counts[i].name,
                g_counts[i].count);
        first = 0;
    }
    fprintf(g_out, "}}\n");
    fflush(g_out);
    pthread_mutex_unlock(&g_lock);
}

/* severity: "blocker" resources cannot be checkpointed at all;
 * "conditional" ones are fine only under stated conditions. */
static void record(const char *api, const char *severity, const char *detail)
{
    if (!g_ready)
        return;
    bump(api);
    if (!g_out)
        return;

    pthread_mutex_lock(&g_lock);
    fprintf(g_out,
            "{\"event\":\"call\",\"pid\":%d,\"tid\":%ld,\"api\":\"%s\","
            "\"severity\":\"%s\",\"detail\":\"%s\"",
            (int)getpid(), (long)pthread_self(), api, severity,
            detail ? detail : "");

#ifdef HAVE_BACKTRACE
    if (g_backtrace) {
        void *frames[MAX_FRAMES];
        int n = backtrace(frames, MAX_FRAMES);
        char **syms = backtrace_symbols(frames, n);
        fprintf(g_out, ",\"stack\":[");
        for (int i = 1; i < n && i < MAX_FRAMES; i++) {
            /* Quotes and backslashes would break the line; symbol text is
             * ASCII-ish but not guaranteed clean. */
            fprintf(g_out, "%s\"", i == 1 ? "" : ",");
            for (const char *p = syms ? syms[i] : "?"; *p; p++) {
                if (*p == '"' || *p == '\\')
                    fputc('\\', g_out);
                fputc(*p, g_out);
            }
            fputc('"', g_out);
        }
        fprintf(g_out, "]");
        free(syms);
    }
#endif

    fprintf(g_out, "}\n");
    fflush(g_out);
    pthread_mutex_unlock(&g_lock);
}

__attribute__((constructor)) static void audit_init(void)
{
    const char *path = getenv("MNCR_AUDIT_OUT");
    g_backtrace = getenv("MNCR_AUDIT_STACKS") != NULL;
    if (path && *path) {
        char resolved[4096];
        /* One file per pid keeps concurrent ranks from interleaving lines. */
        snprintf(resolved, sizeof resolved, "%s.%d", path, (int)getpid());
        g_out = fopen(resolved, "we");
    }
    if (!g_out)
        g_out = stderr;
    g_ready = 1;
    atexit(emit_summary);
}

/* ----------------------------------------------------------------- helpers */

static void *real_sym(const char *name)
{
    /* RTLD_NEXT covers the LD_PRELOAD case, which is how this runs in
     * production. The explicit dlopen is the fallback for everything else: a
     * libcuda at a non-standard path, and the test rig, which points
     * MNCR_AUDIT_REAL_LIB at a stand-in so the interposer can be exercised
     * without a driver. */
    void *fn = dlsym(RTLD_NEXT, name);
    if (!fn) {
        static void *libcuda;
        static int tried;
        if (!tried) {
            const char *path = getenv("MNCR_AUDIT_REAL_LIB");
            libcuda = dlopen(path ? path : "libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
            tried = 1;
            if (!libcuda && g_out) {
                pthread_mutex_lock(&g_lock);
                fprintf(g_out,
                        "{\"event\":\"error\",\"message\":\"cannot resolve the "
                        "real CUDA entry points\",\"tried\":\"%s\"}\n",
                        path ? path : "libcuda.so.1");
                fflush(g_out);
                pthread_mutex_unlock(&g_lock);
            }
        }
        if (libcuda)
            fn = dlsym(libcuda, name);
    }
    return fn;
}

#define REAL(name, type)                                                       \
    static type real;                                                          \
    if (!real)                                                                 \
        real = (type)real_sym(name);                                           \
    if (!real)                                                                 \
        return -1 /* CUDA_ERROR_INVALID_VALUE-ish; the app will surface it */

/* --------------------------------------------------------------- intercepts */

typedef CUresult (*fn_memcreate)(CUmemGenericAllocationHandle *, size_t,
                                 const void *, unsigned long long);
CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t size,
                     const void *prop, unsigned long long flags)
{
    REAL("cuMemCreate", fn_memcreate);
    char detail[64];
    /* Measured on 595: holding VMM allocations does not stop a checkpoint.
     * Recorded because it is worth knowing which process has them - the
     * blocker is importing someone else's, not creating your own. */
    snprintf(detail, sizeof detail, "bytes=%zu", size);
    record("cuMemCreate", "conditional", detail);
    return real(h, size, prop, flags);
}

typedef CUresult (*fn_memmap)(CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle, unsigned long long);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle h, unsigned long long flags)
{
    REAL("cuMemMap", fn_memmap);
    record("cuMemMap", "conditional", "vmm mapping");
    return real(ptr, size, offset, h, flags);
}

typedef CUresult (*fn_export)(void *, CUmemGenericAllocationHandle, int,
                              unsigned long long);
CUresult cuMemExportToShareableHandle(void *out, CUmemGenericAllocationHandle h,
                                      int handle_type, unsigned long long flags)
{
    REAL("cuMemExportToShareableHandle", fn_export);
    char detail[64];
    /* handle type 0x1 = POSIX fd, 0x8 = FABRIC in current headers. Recorded
     * numerically so the report can name it without us guessing here.
     *
     * Measured on 595: the exporting process checkpoints and restores fine
     * even while a peer maps the memory. It is the importer that cannot be
     * restored. So this is conditional, and the import below is the blocker.
     * See docs/findings-595-blackwell.md. */
    snprintf(detail, sizeof detail, "handle_type=%d", handle_type);
    record("cuMemExportToShareableHandle", "conditional", detail);
    return real(out, h, handle_type, flags);
}

typedef CUresult (*fn_import)(CUmemGenericAllocationHandle *, void *, int);
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *h,
                                        void *osh, int handle_type)
{
    REAL("cuMemImportFromShareableHandle", fn_import);
    char detail[64];
    /* The measured failure mode, and it is the worst kind: the checkpoint
     * succeeds and the restore does not, past the point where anything can be
     * undone. */
    snprintf(detail, sizeof detail, "handle_type=%d", handle_type);
    record("cuMemImportFromShareableHandle", "blocker", detail);
    return real(h, osh, handle_type);
}

typedef CUresult (*fn_managed)(CUdeviceptr *, size_t, unsigned int);
CUresult cuMemAllocManaged(CUdeviceptr *ptr, size_t size, unsigned int flags)
{
    REAL("cuMemAllocManaged", fn_managed);
    char detail[64];
    snprintf(detail, sizeof detail, "bytes=%zu", size);
    record("cuMemAllocManaged", "blocker", detail);
    return real(ptr, size, flags);
}

typedef CUresult (*fn_mccreate)(CUmemGenericAllocationHandle *, const void *);
CUresult cuMulticastCreate(CUmemGenericAllocationHandle *h, const void *prop)
{
    REAL("cuMulticastCreate", fn_mccreate);
    record("cuMulticastCreate", "blocker", "nvls multicast object");
    return real(h, prop);
}

typedef CUresult (*fn_mcbindmem)(CUmemGenericAllocationHandle, size_t,
                                 CUmemGenericAllocationHandle, size_t, size_t,
                                 unsigned long long);
CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mcoff,
                            CUmemGenericAllocationHandle mem, size_t memoff,
                            size_t size, unsigned long long flags)
{
    REAL("cuMulticastBindMem", fn_mcbindmem);
    record("cuMulticastBindMem", "blocker", "nvls binding");
    return real(mc, mcoff, mem, memoff, size, flags);
}

typedef CUresult (*fn_ipcget)(CUipcMemHandle *, CUdeviceptr);
CUresult cuIpcGetMemHandle(CUipcMemHandle *h, CUdeviceptr ptr)
{
    REAL("cuIpcGetMemHandle", fn_ipcget);
    record("cuIpcGetMemHandle", "conditional", "needs driver 610 + job file");
    return real(h, ptr);
}

typedef CUresult (*fn_ipcopen)(CUdeviceptr *, CUipcMemHandle, unsigned int);
CUresult cuIpcOpenMemHandle(CUdeviceptr *ptr, CUipcMemHandle h, unsigned int flags)
{
    REAL("cuIpcOpenMemHandle", fn_ipcopen);
    record("cuIpcOpenMemHandle", "conditional", "needs driver 610 + job file");
    return real(ptr, h, flags);
}

/* ------------------------------------------------ cuGetProcAddress redirect */

struct redirect {
    const char *name;
    void *wrapper;
};

static const struct redirect g_redirects[] = {
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

static void *redirect_for(const char *symbol)
{
    if (!symbol)
        return NULL;
    for (size_t i = 0; i < sizeof g_redirects / sizeof g_redirects[0]; i++) {
        if (strcmp(g_redirects[i].name, symbol) == 0)
            return g_redirects[i].wrapper;
    }
    return NULL;
}

typedef CUresult (*fn_getproc)(const char *, void **, int, uint64_t);
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cuda_version,
                          uint64_t flags)
{
    REAL("cuGetProcAddress", fn_getproc);
    CUresult rc = real(symbol, pfn, cuda_version, flags);
    void *ours = redirect_for(symbol);
    if (rc == 0 && ours && pfn) {
        record("cuGetProcAddress", "info", symbol);
        *pfn = ours;
    }
    return rc;
}

typedef CUresult (*fn_getproc2)(const char *, void **, int, uint64_t, void *);
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cuda_version,
                             uint64_t flags, void *status)
{
    REAL("cuGetProcAddress_v2", fn_getproc2);
    CUresult rc = real(symbol, pfn, cuda_version, flags, status);
    void *ours = redirect_for(symbol);
    if (rc == 0 && ours && pfn) {
        record("cuGetProcAddress", "info", symbol);
        *pfn = ours;
    }
    return rc;
}
