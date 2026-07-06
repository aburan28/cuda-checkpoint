/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 *
 * Checkpoint and Restore GPU Migration Demo - CUDA API
 * Requires display driver 580 or higher
 * Run with persistence mode enabled
 *
 * Build with the CUDA 13.0 (or later) toolkit as follows:
 * gcc -I /usr/local/cuda-13.0/include r580-migration-api.c -o r580-migration-api -lcuda -lnvidia-ml
 */

#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda.h>
#include <nvml.h>

#define CHECK(x) check((x) != 0, #x, __FILE__, __LINE__)
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_OK(x) CHECK_EQ(x, 0)

void check(bool is_ok, const char *msg, const char *file, int line)
{
    if (!is_ok) {
        fprintf(stderr, "Error: \"%s\" failed at %s:%d\n", msg, file, line);
        abort();
    }
}

void check_persistence_mode(void)
{
    nvmlDevice_t dev;
    nvmlEnableState_t persistence_mode;

    CHECK_OK(nvmlInit());
    CHECK_OK(nvmlDeviceGetHandleByIndex(0, &dev));
    CHECK_OK(nvmlDeviceGetPersistenceMode(dev, &persistence_mode));
    CHECK_EQ(persistence_mode, NVML_FEATURE_ENABLED);
}

void print_current_ctx_uuid(void)
{
    CUdevice dev;
    CUuuid uuid;

    CHECK_OK(cuCtxGetDevice(&dev));
    CHECK_OK(cuDeviceGetUuid(&uuid, dev));

    for (int i = 0; i < sizeof uuid; i++) {
        printf("%02x", uuid.bytes[i] & 0xFF);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    CHECK_OK(unsetenv("CUDA_VISIBLE_DEVICES"));
    CHECK_OK(unsetenv("CUDA_DEVICE_ORDER"));

    check_persistence_mode();

    CUcontext ctx;
    CHECK_OK(cuInit(0));
    CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_OK(cuCtxSetCurrent(ctx));

    int dev_count;
    CHECK_OK(cuDeviceGetCount(&dev_count));

    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};

    CUcheckpointGpuPair *pairs = calloc(dev_count, sizeof pairs[0]);

    // Upon every restore, move everything on GPU i to GPU i + 1 (with wrap-around)
    // Every GPU accessible to CUDA must be specified, even if the application doesn't use them
    for (int i = 0; i < dev_count; i++) {
        CHECK_OK(cuDeviceGetUuid(&pairs[i].oldUuid, i));
        CHECK_OK(cuDeviceGetUuid(&pairs[i].newUuid, (i + 1) % dev_count));
    }

    // assign the GPU pairs to the restore args
    restore_args.gpuPairsCount = dev_count;
    restore_args.gpuPairs = pairs;

    pid_t self = getpid();
    for (int i = 0; i < dev_count; i++) {
        print_current_ctx_uuid();
        CHECK_OK(cuCheckpointProcessLock(self, &lock_args));
        CHECK_OK(cuCheckpointProcessCheckpoint(self, &checkpoint_args));
        CHECK_OK(cuCheckpointProcessRestore(self, &restore_args));
        CHECK_OK(cuCheckpointProcessUnlock(self, &unlock_args));
    }

    print_current_ctx_uuid();
    return 0;
}
