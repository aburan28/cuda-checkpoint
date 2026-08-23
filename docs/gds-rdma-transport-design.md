# External Transport Design for CUDA Checkpoint/Restore

Status: implementation proposal

Related issues: [#14](https://github.com/NVIDIA/cuda-checkpoint/issues/14),
[#33](https://github.com/NVIDIA/cuda-checkpoint/issues/33),
[#38](https://github.com/NVIDIA/cuda-checkpoint/issues/38), and
[#49](https://github.com/NVIDIA/cuda-checkpoint/issues/49)

## Summary

CUDA checkpoint currently copies device allocations into driver-managed host
memory. This proposal adds a versioned, transport-neutral storage descriptor to
the existing checkpoint and restore calls. The first backend accepts a regular
file descriptor and uses a direct GPU-to-storage path when the file and platform
support GPUDirect Storage (GDS). The same file-descriptor backend also covers
GDS-supported network filesystems, including RDMA-backed filesystems, without
making the CUDA API depend on a particular filesystem or RDMA stack.

The design has four non-negotiable properties:

1. Existing zero-initialized argument structures keep their current behavior.
2. A caller can require a direct path and receive an error instead of silently
   falling back to host staging.
3. Successful checkpoint completion has precisely defined durability semantics.
4. A transport failure does not discard the process's only valid GPU state.

This document describes a driver-facing design. The checkpoint data path is
implemented by the NVIDIA display driver and is not present in this repository.
The benchmark in `src/gds-transport-benchmark.cu` measures the potential data
movement improvement but does not alter `cuda-checkpoint` itself.

## Goals and non-goals

### Goals

- Remove the full-size host-memory copy for device allocations when a direct
  transport is available.
- Support checkpoint and restore through an application-supplied file
  descriptor.
- Support local GDS storage and RDMA-backed GDS filesystems through one API.
- Leave room for non-filesystem RDMA providers without committing the public API
  to verbs, UCX, libfabric, or another specific userspace stack.
- Report which path was actually used and how many bytes used each path.
- Preserve GPU migration through the existing `gpuPairs` restore mapping.

### Non-goals for the first implementation

- Exposing target-process device virtual addresses to the controlling process.
- Exposing internal driver host buffers.
- Defining a general CUDA allocation serialization format for applications.
- Transparent encryption or compression in the direct path. Those operations
  require a separately designed GPU transform pipeline.
- Restoring application network connections. CRIU or another CPU checkpointing
  system remains responsible for CPU and network state.

## Why the integration belongs in the driver

cuFile device pointers are interpreted in the CUDA context of the caller. The
process invoking `cuCheckpointProcessCheckpoint` does not own the target
process's CUDA virtual addresses, and the target's GPU allocations are released
as part of checkpoint. A userspace wrapper therefore cannot safely insert
`cuFileWrite` between the existing checkpoint API and its hidden host buffers.

The checkpoint engine, however, already owns the following information:

- every allocation's backing storage and valid byte ranges;
- the ordering required to quiesce CUDA work;
- allocation-to-GPU identity and migration metadata;
- the point at which an allocation can safely be released; and
- the point at which restored bytes may become visible to CUDA work.

That engine must submit direct I/O or hand registered GPU extents to an internal
transport backend.

## Proposed public API

The names below are illustrative. Normal CUDA API review and symbol versioning
rules take precedence.

```c
typedef enum CUcheckpointStorageType_enum {
    CU_CHECKPOINT_STORAGE_HOST = 0,
    CU_CHECKPOINT_STORAGE_FD   = 1,
    CU_CHECKPOINT_STORAGE_PROVIDER_FD = 2
} CUcheckpointStorageType;

typedef enum CUcheckpointStorageFlags_enum {
    /* Fail if any device payload byte would pass through host memory. */
    CU_CHECKPOINT_STORAGE_DIRECT_REQUIRED = 1u << 0,

    /* Permit bounded host bounce buffers and report their use in result. */
    CU_CHECKPOINT_STORAGE_ALLOW_FALLBACK  = 1u << 1,

    /* Checkpoint returns only after payload and commit record are durable. */
    CU_CHECKPOINT_STORAGE_DURABLE         = 1u << 2,

    /* The supplied range may be overwritten by this checkpoint. */
    CU_CHECKPOINT_STORAGE_OVERWRITE       = 1u << 3
} CUcheckpointStorageFlags;

typedef struct CUcheckpointStorageFd_st {
    unsigned int structSize;
    unsigned int flags;
    int fd;
    unsigned int reserved0;
    cuuint64_t offset;
    cuuint64_t capacity;
    cuuint64_t reserved[4];
} CUcheckpointStorageFd;

typedef struct CUcheckpointStorageDesc_st {
    unsigned int structSize;
    unsigned int version;
    CUcheckpointStorageType type;
    unsigned int reserved0;
    union {
        CUcheckpointStorageFd file;
        cuuint64_t reserved[8];
    } handle;
} CUcheckpointStorageDesc;

typedef enum CUcheckpointDataPath_enum {
    CU_CHECKPOINT_DATA_PATH_NONE         = 0,
    CU_CHECKPOINT_DATA_PATH_HOST         = 1,
    CU_CHECKPOINT_DATA_PATH_GDS          = 2,
    CU_CHECKPOINT_DATA_PATH_GDS_RDMA     = 3,
    CU_CHECKPOINT_DATA_PATH_PROVIDER     = 4,
    CU_CHECKPOINT_DATA_PATH_MIXED        = 5
} CUcheckpointDataPath;

typedef struct CUcheckpointStorageResult_st {
    unsigned int structSize;
    unsigned int version;
    CUcheckpointDataPath dataPath;
    unsigned int reserved0;
    cuuint64_t imageBytes;
    cuuint64_t directBytes;
    cuuint64_t fallbackBytes;
    cuuint64_t committedOffset;
    cuuint64_t reserved[4];
} CUcheckpointStorageResult;

typedef struct CUcheckpointStorageCaps_st {
    unsigned int structSize;
    unsigned int version;
    unsigned int supportedFlags;
    unsigned int directSupported;
    cuuint64_t requiredOffsetAlignment;
    cuuint64_t requiredSizeAlignment;
    cuuint64_t requiredDeviceAlignment;
    cuuint64_t reserved[4];
} CUcheckpointStorageCaps;

CUresult cuCheckpointProcessQueryStorage(
    int pid,
    const CUcheckpointStorageDesc *storage,
    CUcheckpointStorageCaps *capabilities);
```

The query is a non-destructive preflight against the target process, its GPUs,
and the supplied endpoint. Applications should resolve it with
`cuGetProcAddress`; absence of the symbol means the installed driver predates
external checkpoint storage. Endpoint state may still change after a successful
query, so checkpoint and restore repeat all safety checks.

The existing 64-byte argument structures can consume currently reserved space
without changing their total size. Conceptually, the new fields are:

```c
typedef struct CUcheckpointCheckpointArgs_st {
    const CUcheckpointStorageDesc *storage;
    CUcheckpointStorageResult *storageResult;
    cuuint64_t reserved[6];
} CUcheckpointCheckpointArgs;

typedef struct CUcheckpointRestoreArgs_st {
    CUcheckpointGpuPair *gpuPairs;
    unsigned int gpuPairsCount;
    unsigned int reserved0;
    const CUcheckpointStorageDesc *storage;
    CUcheckpointStorageResult *storageResult;
    cuuint64_t reserved[4];
} CUcheckpointRestoreArgs;

static_assert(sizeof(CUcheckpointCheckpointArgs) == 64, "checkpoint ABI");
static_assert(sizeof(CUcheckpointRestoreArgs) == 64, "restore ABI");
```

This restore layout is for the currently supported 64-bit x86 and Arm ABIs. The
new `reserved0` occupies bytes that legacy headers require callers to zero, so
the offsets of `gpuPairs` and `gpuPairsCount` remain unchanged. Compile-time size
and field-offset assertions are required on every supported architecture.

Legacy callers zero the structures, so `storage == NULL` selects the existing
host-memory behavior. Every pointed-to structure begins with `structSize` and a
version, allowing fields to be appended without reading beyond a caller's
allocation. Unknown flags, nonzero reserved fields, invalid flag combinations,
and unsupported versions return `CUDA_ERROR_INVALID_VALUE`.

Backward compatibility is directional: old applications work unchanged on a
new driver. A new application must use the query symbol before placing nonzero
values in fields that an old driver still considers reserved. If it ignores that
rule, an old driver is expected to reject the arguments with
`CUDA_ERROR_INVALID_VALUE`.

### File descriptor ownership

- `fd` belongs to the process calling the checkpoint API, not the target CUDA
  process.
- The driver resolves the descriptor during the API call and holds a kernel file
  reference until the synchronous operation completes.
- Closing or reusing the numeric descriptor concurrently is a caller error, but
  cannot redirect an operation after the driver has acquired its reference.
- `offset` and `capacity` delimit the only byte range the driver may access.
- The driver does not truncate the file and never reads or writes outside that
  range.
- Checkpoint and restore may use different numeric descriptors as long as they
  refer to the same completed image bytes.

### Directness and fallback

`DIRECT_REQUIRED` and `ALLOW_FALLBACK` are mutually exclusive. If neither is
specified, the initial API should behave as `DIRECT_REQUIRED`; silent fallback
would make the feature impossible to operate or benchmark reliably.

Directness applies to device payload bytes. Small metadata records may originate
in host memory, but `fallbackBytes` must remain zero unless device payload bytes
were staged through host DRAM. A mixed result is legal only when fallback was
explicitly allowed.

The result structure is written on both success and recoverable failure whenever
its address and size are valid. This lets callers distinguish unsupported
storage, insufficient capacity, partial direct-path support, and a transport
error after some bytes were submitted.

## Image layout and atomic completion

The external image is a driver-private, versioned container. It should not expose
raw kernel pointers or require CRIU to interpret allocation records.

```text
range offset
    │
    ▼
+----------------------+  fixed header, image UUID, format version
| provisional header   |
+----------------------+  GPU and allocation table
| manifest             |
+----------------------+  aligned device payload extents
| payload 0            |
| payload 1            |
| ...                  |
+----------------------+  hashes and final lengths
| completed manifest   |
+----------------------+  written last
| commit record        |
+----------------------+
```

The provisional header is not sufficient to restore an image. A restore accepts
only a validated commit record whose image UUID, manifest hash, byte counts, and
format version agree. The commit record is written after every payload write has
completed. With `DURABLE`, the driver also issues the filesystem/backend flush
needed to make the entire range durable before returning success.

Payload offsets and lengths are aligned to the backend's direct-I/O constraints.
Padding bytes are included in `imageBytes` but not allocation sizes. The manifest
records the source GPU UUID and allocation metadata needed by the existing
restore and GPU-remapping logic.

A checksum is required for metadata. Per-extent payload checksums are strongly
recommended; they may be calculated on the GPU to preserve a direct data path.
The checksum algorithm and coverage are format fields, not implicit driver
behavior.

The checkpoint engine also retains the external image UUID and expected commit
identity in the target process's CPU-visible checkpoint metadata. CRIU persists
that small record with the process. Restore rejects a valid but unrelated image,
rather than accepting any container that happens to have a valid commit record.

## State machine and failure behavior

The existing externally visible process states remain unchanged:

```text
RUNNING --lock--> LOCKED --checkpoint--> CHECKPOINTED
   ^                 |                         |
   |                 +--unlock after error----+
   |                                           |
   +------------unlock<--LOCKED<--restore------+
```

Checkpoint ordering:

1. Validate and acquire the storage endpoint.
2. Determine capacity and direct-path support before destructive work.
3. Quiesce the target as the existing checkpoint implementation does.
4. Emit provisional metadata.
5. Transfer each allocation. Do not release an allocation until its I/O has
   completed successfully and any required integrity metadata is retained.
6. Emit and, when requested, durably flush the commit record.
7. Release remaining GPU resources and transition to `CHECKPOINTED`.

If any step before the commit record fails, the image remains uncommitted. The
process remains `LOCKED`, and the driver must retain or reconstruct enough GPU
state for the caller to retry checkpoint or unlock. An implementation must not
release the only valid copy of an allocation before a failed write can be
recovered. A practical first version may serialize allocation writes and defer
GPU release until the whole image commits, trading peak GPU reclamation for a
clear failure guarantee.

Restore ordering:

1. Validate the commit record and complete manifest before allocating GPUs.
2. Validate format compatibility, capacity, checksums, and all `gpuPairs`.
3. Allocate and map destination GPU memory.
4. Transfer payload extents directly to the destination allocations.
5. Restore CUDA objects only after all payload reads and integrity checks pass.
6. Transition from `CHECKPOINTED` to `LOCKED`; the existing unlock call resumes
   CUDA work.

A restore failure leaves the process `CHECKPOINTED` and eligible for retry. No
CUDA thread may observe a partially restored allocation.

## Backend architecture

The checkpoint engine should depend on a small internal interface rather than on
cuFile or RDMA-specific types:

```c
struct checkpoint_transport_ops {
    int  (*open)(struct checkpoint_transport *, const struct endpoint *);
    int  (*query)(struct checkpoint_transport *, struct capabilities *);
    int  (*write_gpu)(struct checkpoint_transport *, struct gpu_extent *,
                      u64 storage_offset, u64 length);
    int  (*read_gpu)(struct checkpoint_transport *, struct gpu_extent *,
                     u64 storage_offset, u64 length);
    int  (*write_metadata)(struct checkpoint_transport *, const void *,
                           u64 storage_offset, u64 length);
    int  (*flush)(struct checkpoint_transport *);
    void (*close)(struct checkpoint_transport *);
};
```

The operations are synchronous at this layer; backends may queue multiple
requests internally. Completion must be joined before an extent is released or
made visible.

### Host backend

The current implementation becomes the default backend when `storage == NULL`.
It is also the bounded fallback backend. Fallback uses a configurable chunked
bounce buffer rather than allocating host memory equal to total GPU memory.

### File/GDS backend

The endpoint is a referenced regular-file descriptor plus an allowed byte range.
During `query`, the backend verifies:

- readable/writable mode appropriate to the requested operation;
- filesystem and device support for direct GPU I/O;
- offset, length, and device-address alignment requirements;
- GPU/backend topology and peer-DMA reachability; and
- whether all allocations can use the direct path.

The driver should reuse supported NVIDIA GDS kernel interfaces rather than call
the userspace `libcufile` API from kernel context. GDS-backed network
filesystems select their RDMA path internally, producing a
`GDS_RDMA` result when that route can be identified reliably.

### Provider-fd backend

`PROVIDER_FD` is reserved for a future kernel endpoint that implements a stable
CUDA checkpoint transport protocol over a file descriptor. The protocol would
negotiate capabilities, register GPU extents, submit reads/writes, and report
completion. This accommodates future RDMA, fabric, or object-storage providers
without embedding provider-specific structures in the CUDA API.

It should not be exposed publicly until there is at least one implementation and
a security review of endpoint lifetime, DMA authorization, IOMMU isolation, and
target-process ownership.

## Security requirements

- Apply normal CUDA checkpoint permission checks before resolving the endpoint.
- Never trust lengths, offsets, completion values, or capability flags returned
  by a provider; validate overflow and bounds in the driver.
- Pin endpoint identity after descriptor resolution to prevent fd-reuse races.
- Bind every registered GPU extent to the target process and checkpoint
  operation; revoke it on completion or error.
- Zero padding and metadata-reserved fields before writing them.
- Do not serialize kernel pointers, stale GPU virtual addresses, or uninitialized
  driver memory.
- Treat checkpoint images as sensitive application memory. Permissions,
  encryption at rest, and secure deletion remain caller/storage responsibilities
  unless a later encrypted-image feature explicitly provides them.
- Validate image metadata and all arithmetic before allocating destination GPU
  resources during restore.

## Observability

In addition to `CUcheckpointStorageResult`, driver debug counters should expose:

- direct, fallback, metadata, and padding bytes;
- submission and completion latency;
- flush latency;
- backend and route selection;
- registration-cache hits/misses;
- retry and short-I/O counts; and
- the first backend error without logging file contents or application data.

The CLI can later expose these values through an opt-in machine-readable output
flag. It must not infer GDS usage merely from a successful file-backed
checkpoint.

## Proposed CLI surface

After the driver API exists, the utility can add the following options without
changing the existing commands:

```text
--storage-file <path>
        Store or restore device payloads in the specified external image.

--storage-offset <bytes>
        Start of the permitted image range. Default: 0.

--storage-capacity <bytes>
        Size of the permitted image range. Required for checkpoint.

--storage-mode direct | fallback
        Require a direct GPU path or explicitly permit bounded host staging.
        Default: direct.

--durable
        Do not complete checkpoint until the external image is durable.

--storage-stats
        Print the selected data path and byte counters as JSON on stderr.
```

For checkpoint, the CLI opens an existing, caller-sized file read/write with
direct-I/O intent and never implicitly truncates bytes outside the permitted
range. For restore it opens the file read-only. Separate `--action checkpoint`
and `--action restore` invocations may open the same path with different numeric
file descriptors. `--toggle` accepts these options but requires a storage file
whenever the current state implies restore from an external image.

Examples:

```bash
truncate -s 1T /mnt/gds/job-42.cuda-image

cuda-checkpoint --action checkpoint --pid "$PID" \
    --storage-file /mnt/gds/job-42.cuda-image \
    --storage-capacity 1T --storage-mode direct --durable --storage-stats

cuda-checkpoint --action restore --pid "$PID" \
    --storage-file /mnt/gds/job-42.cuda-image \
    --storage-mode direct --storage-stats
```

## Implementation sequence

1. Refactor the existing host copy into `checkpoint_transport_ops` without a
   behavior change.
2. Add descriptor parsing, ABI assertions, result reporting, and negative tests.
3. Implement an fd backend using bounded host I/O. Gate it behind
   `ALLOW_FALLBACK`; this validates the image format and retry behavior.
4. Implement direct GPU extent registration and I/O for supported local GDS
   filesystems.
5. Add GDS-backed RDMA filesystem detection and reporting.
6. Optimize queue depth, registration caching, allocation release timing, and
   GPU-side checksums.
7. Evaluate the provider-fd interface only after the fd/GDS API has stabilized.

## Required tests

### ABI and validation

- Legacy zeroed arguments on old and new drivers.
- Capability-query symbol discovery and new headers against an old driver.
- Structure size/offset assertions on x86-64 and aarch64.
- Unknown versions, flags, storage types, and nonzero reserved fields.
- Invalid descriptors, modes, ranges, overflows, and insufficient capacity.

### Correctness

- Single and multiple GPUs, allocations, contexts, and sparse mappings.
- Zero-byte and alignment-boundary extents.
- GPU remapping on restore.
- Repeated checkpoint/restore cycles with byte-for-byte device validation.
- CRIU process-tree dump/restore using an external image descriptor.
- Driver/toolkit version compatibility and intentionally corrupted manifests,
  payloads, checksums, and commit records.

### Failure injection

- Short I/O and errors before, during, and after each allocation.
- ENOSPC, endpoint revocation, filesystem unmount, RDMA disconnect, GPU reset,
  flush failure, and controller-process termination.
- Retry checkpoint, retry restore, and unlock after every recoverable failure.
- Confirm that an uncommitted image is never accepted.

### Performance

- Report checkpoint latency, restore latency, time to durable completion, CPU
  utilization, peak host RSS, GPU memory release time, and effective bandwidth.
- Compare current host staging, bounded fallback, local GDS, and GDS over RDMA.
- Test cold and warm registration caches, multiple allocation sizes, queue
  depths, NUMA placements, and multi-GPU concurrency.
- Assert `fallbackBytes == 0` for every direct-required performance run.

The repository benchmark provides a baseline for the two data-movement shapes.
End-to-end validation still requires a driver containing the proposed backend.
