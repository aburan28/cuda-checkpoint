/*
 * Calls every hooked entry point two ways: directly, and through the pointer
 * cuGetProcAddress hands back. Both must show up in the audit log.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int CUresult;

extern CUresult cuMemCreate(void *, size_t, const void *, unsigned long long);
extern CUresult cuMemMap(unsigned long long, size_t, size_t, void *, unsigned long long);
extern CUresult cuMemExportToShareableHandle(void *, void *, int, unsigned long long);
extern CUresult cuMemAllocManaged(unsigned long long *, size_t, unsigned int);
extern CUresult cuMulticastCreate(void *, const void *);
extern CUresult cuIpcGetMemHandle(void *, unsigned long long);
extern CUresult cuGetProcAddress(const char *, void **, int, uint64_t);

typedef CUresult (*fn_create)(void *, size_t, const void *, unsigned long long);
typedef CUresult (*fn_export)(void *, void *, int, unsigned long long);
typedef CUresult (*fn_getproc)(const char *, void **, int, uint64_t);

int main(void)
{
    void *handle = NULL;
    unsigned long long ptr = 0;
    char blob[64] = {0};

    /* direct calls - the LD_PRELOAD path */
    cuMemCreate(&handle, 1u << 20, NULL, 0);
    cuMemMap(ptr, 1u << 20, 0, handle, 0);
    cuMemExportToShareableHandle(blob, handle, 1, 0);
    cuMemAllocManaged(&ptr, 4096, 1);
    cuMulticastCreate(&handle, NULL);
    cuIpcGetMemHandle(blob, ptr);

    /* resolved calls - the cuGetProcAddress path, which is what NCCL uses */
    fn_create resolved_create = NULL;
    if (cuGetProcAddress("cuMemCreate", (void **)&resolved_create, 12080, 0) != 0) {
        fprintf(stderr, "cuGetProcAddress(cuMemCreate) failed\n");
        return 1;
    }
    resolved_create(&handle, 2u << 20, NULL, 0);

    fn_export resolved_export = NULL;
    if (cuGetProcAddress("cuMemExportToShareableHandle", (void **)&resolved_export,
                         12080, 0) != 0) {
        fprintf(stderr, "cuGetProcAddress(cuMemExportToShareableHandle) failed\n");
        return 1;
    }
    /* handle type 8 is FABRIC in current headers: the MNNVL case */
    resolved_export(blob, handle, 8, 0);

    /* dlopen + dlsym - how PyTorch and NCCL actually reach libcuda, and the
     * bypass that LD_PRELOAD symbol interposition does not cover on its own. */
    void *lib = dlopen("libfakecuda" LIBSUFFIX, RTLD_LAZY);
    if (lib) {
        fn_create via_dlsym = (fn_create)dlsym(lib, "cuMemCreate");
        if (via_dlsym) {
            via_dlsym(&handle, 3u << 20, NULL, 0);
        } else {
            fprintf(stderr, "dlsym(cuMemCreate) returned NULL\n");
            return 1;
        }

        /* What PyTorch actually does: dlsym the resolver once, then resolve
         * everything else through it. If the resolver handed back is the
         * driver's, every later entry point bypasses the audit. */
        fn_getproc getproc = (fn_getproc)dlsym(lib, "cuGetProcAddress");
        fn_create via_getproc = NULL;
        if (!getproc ||
            getproc("cuMemCreate", (void **)&via_getproc, 12080, 0) != 0 ||
            !via_getproc) {
            fprintf(stderr, "dlsym(cuGetProcAddress) chain failed\n");
            return 1;
        }
        via_getproc(&handle, 4u << 20, NULL, 0);
    }

    printf("probe done\n");
    return 0;
}
