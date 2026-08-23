/*
 * vmm_ipc_probe - is VMM memory that is actually *shared* checkpointable?
 *
 * vmm_probe showed that exporting a shareable handle does not by itself stop a
 * checkpoint. But nothing imported that handle, and the documented limitation
 * is about IPC memory - memory two processes are using. NCCL does exactly that
 * across ranks, so the distinction decides whether communicator teardown is
 * required or merely tidy.
 *
 * Parent creates a VMM allocation, exports a POSIX fd, and passes it to a
 * forked child over SCM_RIGHTS. The child imports and maps it. Both then wait,
 * sharing one physical allocation, for the harness to checkpoint them.
 *
 * Build: nvcc -O2 vmm_ipc_probe.cu -o vmm_ipc_probe -lcuda
 * Run:   ./vmm_ipc_probe <state-prefix>
 */

#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <sys/socket.h>
#include <sys/wait.h>
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

static void announce(const char *path, const char *who)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *fh = fopen(tmp, "w");
    if (!fh)
        return;
    fprintf(fh, "%s %d\n", who, (int)getpid());
    fclose(fh);
    rename(tmp, path);
}

static void wait_for_command(const char *path)
{
    for (int i = 0; i < 1800; i++) {
        char cmd[4096];
        snprintf(cmd, sizeof cmd, "%s.cmd", path);
        FILE *fh = fopen(cmd, "r");
        if (fh) {
            fclose(fh);
            break;
        }
        usleep(100 * 1000);
    }
}

static int send_fd(int sock, int fd)
{
    struct msghdr msg = {};
    char buf[CMSG_SPACE(sizeof(int))] = {};
    char dummy = 'x';
    struct iovec io = {&dummy, 1};

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof buf;

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof fd);

    return sendmsg(sock, &msg, 0) < 0 ? -1 : 0;
}

static int recv_fd(int sock)
{
    struct msghdr msg = {};
    char buf[CMSG_SPACE(sizeof(int))] = {};
    char dummy = 0;
    struct iovec io = {&dummy, 1};

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof buf;

    if (recvmsg(sock, &msg, 0) < 0)
        return -1;
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS)
        return -1;
    int fd = -1;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof fd);
    return fd;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <state-prefix>\n", argv[0]);
        return 2;
    }
    const char *prefix = argv[1];
    char parent_state[4096], child_state[4096];
    snprintf(parent_state, sizeof parent_state, "%s.parent", prefix);
    snprintf(child_state, sizeof child_state, "%s.child", prefix);

    int socks[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, socks) != 0) {
        perror("socketpair");
        return 1;
    }

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 1;
    }

    if (child == 0) {
        close(socks[0]);
        CUdevice dev;
        CUcontext ctx;
        CU(cuInit(0));
        CU(cuDeviceGet(&dev, 0));
        CU(cuDevicePrimaryCtxRetain(&ctx, dev));
        CU(cuCtxSetCurrent(ctx));

        int fd = recv_fd(socks[1]);
        if (fd < 0) {
            fprintf(stderr, "child: no fd received\n");
            return 1;
        }

        CUmemGenericAllocationHandle handle;
        CU(cuMemImportFromShareableHandle(
            &handle, (void *)(uintptr_t)fd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

        size_t size = 2u << 20;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        size_t granularity = 0;
        CU(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        if (granularity > size)
            size = granularity;

        CUdeviceptr ptr;
        CU(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
        CU(cuMemMap(ptr, size, 0, handle, 0));
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU(cuMemSetAccess(ptr, size, &access, 1));

        /* Read what the parent wrote, proving the mapping is live. */
        unsigned char probe = 0;
        CU(cuMemcpyDtoH(&probe, ptr, 1));
        CU(cuCtxSynchronize());

        announce(child_state, probe == 0xEF ? "ready-shared" : "ready-unshared");

        /* Answer verify requests so the harness can ask again after a restore:
         * a checkpoint that succeeds and then hands back corrupt shared memory
         * would be worse than one that refuses. */
        for (int i = 0; i < 3000; i++) {
            char cmd[4096];
            snprintf(cmd, sizeof cmd, "%s.cmd", child_state);
            FILE *fh = fopen(cmd, "r");
            if (fh) {
                char verb[64] = {0};
                if (!fgets(verb, sizeof verb, fh))
                    verb[0] = 0;
                fclose(fh);
                unlink(cmd);
                if (strncmp(verb, "exit", 4) == 0)
                    break;
                probe = 0;
                unsigned char tail = 0;
                if (cuMemcpyDtoH(&probe, ptr, 1) != CUDA_SUCCESS ||
                    cuMemcpyDtoH(&tail, ptr + size - 1, 1) != CUDA_SUCCESS) {
                    announce(child_state, "verify-error");
                } else {
                    announce(child_state,
                             (probe == 0xEF && tail == 0xEF) ? "verify-ok"
                                                             : "verify-BAD");
                }
            }
            usleep(100 * 1000);
        }
        return 0;
    }

    close(socks[1]);
    CUdevice dev;
    CUcontext ctx;
    CU(cuInit(0));
    CU(cuDeviceGet(&dev, 0));
    CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU(cuCtxSetCurrent(ctx));

    size_t size = 2u << 20;
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
    CU(cuCtxSynchronize());

    int fd = -1;
    CU(cuMemExportToShareableHandle(&fd, handle,
                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    if (send_fd(socks[0], fd) != 0) {
        perror("send_fd");
        return 1;
    }

    announce(parent_state, "ready");
    wait_for_command(parent_state);
    kill(child, SIGTERM);
    waitpid(child, NULL, 0);
    return 0;
}
