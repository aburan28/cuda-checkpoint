/*
 * vmm_probe - answers the plan's biggest open question on real hardware.
 *
 * PyTorch's expandable segments allocate through cuMemCreate/cuMemMap. The
 * documented limitation names cuMemExportToShareableHandle - the *export* - not
 * the allocation. Which of those the driver actually refuses decides a
 * fleet-wide policy, and it is not answerable from documentation.
 *
 * Each mode holds one kind of allocation and then waits. The harness
 * checkpoints it from outside and reports whether the driver accepted it.
 *
 *   plain     cuMemAlloc only                        control: must succeed
 *   vmm       cuMemCreate + cuMemMap, not exported   <- the question
 *   exported  vmm + cuMemExportToShareableHandle     expected to fail
 *   managed   cuMemAllocManaged (UVM)                expected to fail
 *   ipc       cuMemAlloc + cuIpcGetMemHandle         610 feature on a 595 node
 *
 * Build: nvcc -O2 vmm_probe.cu -o vmm_probe -lcuda
 * Run:   ./vmm_probe <mode> <state-file>
 */

#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <unistd.h>

#define CU(x)                                                                  \
    do {                                                                       \
        CUresult rc_ = (x);                                                    \
        if (rc_ != CUDA_SUCCESS) {                                             \
            const char *msg = NULL;                                            \
            cuGetErrorString(rc_, &msg);                                       \
            fprintf(stderr, "%s failed: %d %s\n", #x, (int)rc_,                \
                    msg ? msg : "?");                                          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static void announce(const char *path, const char *text)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *fh = fopen(tmp, "w");
    if (!fh)
        return;
    fprintf(fh, "%s %d\n", text, (int)getpid());
    fclose(fh);
    rename(tmp, path);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <plain|vmm|exported|managed|ipc> <state>\n",
                argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    const char *state = argv[2];

    CUdevice dev;
    CUcontext ctx;
    CU(cuInit(0));
    CU(cuDeviceGet(&dev, 0));
    CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU(cuCtxSetCurrent(ctx));

    /* Small on purpose: this node's GPU is nearly full of a live server. */
    size_t size = 2u << 20;

    if (strcmp(mode, "plain") == 0) {
        CUdeviceptr ptr;
        CU(cuMemAlloc(&ptr, size));
        CU(cuMemsetD8(ptr, 0xAB, size));
    } else if (strcmp(mode, "managed") == 0) {
        CUdeviceptr ptr;
        CU(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));
        CU(cuMemsetD8(ptr, 0xCD, size));
    } else if (strcmp(mode, "ipc") == 0) {
        CUdeviceptr ptr;
        CUipcMemHandle handle;
        CU(cuMemAlloc(&ptr, size));
        CU(cuIpcGetMemHandle(&handle, ptr));
    } else {
        /* vmm and exported share the allocation; only the export differs. */
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        size_t granularity = 0;
        CU(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        if (granularity > size)
            size = granularity;

        CUmemGenericAllocationHandle handle;
        CU(cuMemCreate(&handle, size, &prop, 0));

        CUdeviceptr ptr;
        CU(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
        CU(cuMemMap(ptr, size, 0, handle, 0));

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU(cuMemSetAccess(ptr, size, &access, 1));
        CU(cuMemsetD8(ptr, 0xEF, size));

        if (strcmp(mode, "exported") == 0) {
            int fd = -1;
            CU(cuMemExportToShareableHandle(
                &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
            /* Deliberately kept open: the question is whether the driver
             * refuses a process that currently holds an exported handle. */
        }
    }

    CU(cuCtxSynchronize());
    announce(state, "ready");

    /* Wait for the harness. No sockets, nothing to clean up. */
    for (int i = 0; i < 1200; i++) {
        char cmd[4096];
        snprintf(cmd, sizeof cmd, "%s.cmd", state);
        FILE *fh = fopen(cmd, "r");
        if (fh) {
            fclose(fh);
            unlink(cmd);
            break;
        }
        usleep(100 * 1000);
    }
    return 0;
}
