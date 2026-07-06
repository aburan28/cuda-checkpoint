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
 * Checkpoint and Restore GPU Migration Demo - Command Line
 * Requires display driver 580 or higher
 *
 * Build with the CUDA 13.0 (or later) toolkit as follows:
 * gcc -I /usr/local/cuda-13.0/include r580-migration-cli.c -o r580-migration-cli -lcuda
 */

#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>

#define CHECK(x) check((x) != 0, #x, __FILE__, __LINE__)
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_OK(x) CHECK_EQ(x, 0)
#define CHECK_NONNEG(x) CHECK((x) >= 0)

void check(bool is_ok, const char *msg, const char *file, int line)
{
    if (!is_ok) {
        fprintf(stderr, "Error: \"%s\" failed at %s:%d\n", msg, file, line);
        abort();
    }
}

#define CUDA_CHECKPOINT(...) spawn((const char *[]){"cuda-checkpoint", __VA_ARGS__, NULL})
#define UUID_ASCII_SIZE 40

// stores the character sequence "<UUID>=<UUID>," or "<UUID>=<UUID>\0"
typedef struct ascii_uuid_key_value_st
{
    char old_uuid[UUID_ASCII_SIZE];
    char equals;
    char new_uuid[UUID_ASCII_SIZE];
    char comma_or_null;
} ascii_uuid_key_value_t;

void ascii_uuid(char *str, int dev_idx)
{
    CUuuid uuid = {0};
    uint8_t *b = (uint8_t *)uuid.bytes;
    CHECK_OK(cuDeviceGetUuid(&uuid, dev_idx));
    size_t count = sprintf(str,
                           "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                           b[0x0], b[0x1], b[0x2], b[0x3], b[0x4], b[0x5], b[0x6], b[0x7],
                           b[0x8], b[0x9], b[0xA], b[0xB], b[0xC], b[0xD], b[0xE], b[0xF]);
    CHECK_EQ(count, UUID_ASCII_SIZE);
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

void spawn(const char **args)
{
    pid_t child = fork();
    CHECK_NONNEG(child);

    if (child == 0) {
        execvp(args[0], (char * const *)args);
        fprintf(stderr, "Ensure that cuda-checkpoint is in the path and that a 580 or higher driver is installed\n");
        abort();
    }

    int status = 0;
    CHECK_EQ(waitpid(child, &status, 0), child);
    CHECK(WIFEXITED(status));
    CHECK_OK(WEXITSTATUS(status));
}

int main(int argc, char **argv)
{
    CHECK_OK(unsetenv("CUDA_VISIBLE_DEVICES"));
    CHECK_OK(unsetenv("CUDA_DEVICE_ORDER"));

    CUcontext ctx;
    CHECK_OK(cuInit(0));
    CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_OK(cuCtxSetCurrent(ctx));

    int dev_count;
    CHECK_OK(cuDeviceGetCount(&dev_count));

    ascii_uuid_key_value_t *uuid_map = calloc(dev_count, sizeof uuid_map[0]);

    // Upon every restore, move everything on GPU i to GPU i + 1 (with wrap-around)
    // Every GPU accessible to CUDA must be specified, even if the application doesn't use them
    // Build an ASCII string in the format "oldUuid1=newUuid1,oldUuid2=newUuid2,..."
    ascii_uuid_key_value_t *pair = NULL;
    for (int i = 0; i < dev_count; i++) {
        // insert comma after previous entry, if it exists
        if (pair != NULL) {
            pair->comma_or_null = ',';
        }

        // insert GPU UUIDs for current pair
        pair = &uuid_map[i];
        ascii_uuid(pair->old_uuid, i);
        ascii_uuid(pair->new_uuid, (i + 1) % dev_count);
        pair->equals = '=';
    }

    char self[16];
    sprintf(self, "%d", getpid());

    for (int i = 0; i < dev_count; i++) {
        print_current_ctx_uuid();
        CUDA_CHECKPOINT("--action", "lock", "--pid", self);
        CUDA_CHECKPOINT("--action", "checkpoint", "--pid", self);
        CUDA_CHECKPOINT("--action", "restore", "--pid", self, "--device-map", (char *)uuid_map);
        CUDA_CHECKPOINT("--action", "unlock", "--pid", self);
    }

    print_current_ctx_uuid();
    return 0;
}
