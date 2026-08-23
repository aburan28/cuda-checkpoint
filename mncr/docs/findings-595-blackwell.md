# Measured on hardware: driver 595.91.07, RTX PRO 6000 Blackwell Server Edition

Run 2026-08-23 on a g7e.2xlarge (single GPU, 97,887 MiB, persistence enabled),
against `cuda-checkpoint` 595.91.07. Everything below is observed, not inferred.

Reproduce with `verify/vmm_probe.cu` and `verify/vmm_ipc_probe.cu`.

## What checkpoints, and what does not

| Allocation | Checkpoint | Restore | Notes |
|---|---|---|---|
| `cuMemAlloc` | yes | yes | control case; data verified by checksum |
| `cuMemCreate` + `cuMemMap` | **yes** | **yes** | the VMM path, held not shared |
| + `cuMemExportToShareableHandle` (POSIX fd, held open) | **yes** | **yes** | exporting alone is not disqualifying |
| **imported** via `cuMemImportFromShareableHandle` | **yes** | **NO** | `"invalid argument"`, then unrecoverable |
| `cuMemAllocManaged` (UVM) | no | — | `"operation not supported"` |
| `cuIpcGetMemHandle` | no | — | `"OS call failed or operation not supported"`; this is the 610 feature |

## The expandable-segments question, answered

The plan called this its highest-leverage unknown: PyTorch's expandable segments
allocate through `cuMemCreate`/`cuMemMap`, and the documented limitation names
the *export*, not the allocation.

**A process merely holding VMM allocations checkpoints and restores cleanly.**
Expandable segments do not need to be disabled on this driver. The manifests can
stop forcing `expandable_segments:False`.

Caveats, and they matter: driver 595 on a single GPU. Behaviour on 610 was not
measured, and neither was a multi-GPU node.

## The sharper finding: exporting is fine, importing is fatal

The vendor documentation says the utility "does not support UVM memory or IPC
memory created with `cuMemExportToShareableHandle()`". Measured, that statement
splits in two, and the halves behave very differently:

- The process that **exports** a handle checkpoints and restores fine — even
  while another process is actively mapping that memory.
- The process that **imports** it checkpoints fine and then **fails to
  restore**, with `"invalid argument"`. It cannot be unlocked afterwards
  (`"the operation cannot be performed in the present state"`); the process is
  lost.

Tested in both restore orders — same-as-checkpoint and reversed — with identical
results, so this is not the documented ordering requirement. It is the sharing
itself.

### Why this matters more than the table suggests

The failure lands **after the commit point**. The checkpoint succeeds, the
process releases its GPU resources, and only then does restore refuse. There is
no way back for that process — which is exactly the asymmetry this system is
built around, observed in the wild rather than argued from documentation.

For NCCL this settles a design question: ranks import each other's handles, so
the importing side must be torn down before the checkpoint. Communicator
teardown is **required**, not merely tidy.

## Driver 595 versus 610

`cuda-checkpoint` 595.91.07 offers `--device-map` (580+) but not `--launch-job`
in its help, and `cuIpcGetMemHandle` memory is not checkpointable. Single-process
checkpoint and restore works completely. So for a fleet on 595:

- single-process C/R: **supported**
- GPU remap on restore: **supported**
- multi-process legacy IPC: **not until 610**

`agent/preflight.py` treats a driver below 610 as a blocker only when the job
needs job-file IPC, and as a warning otherwise.

## Timing observed

| Operation | Time |
|---|---|
| `lock` | 16 ms |
| `checkpoint` (16 MiB device buffer) | 286 ms |
| GPU released, as seen by `nvidia-smi` | ~1 s **after** the call returns |

That last row is a trap for anyone verifying by hand or in a script: the driver
call returns before `nvidia-smi` stops listing the process. `verify/smoke.py`
polls with a deadline rather than checking once, and an operator should too.

## The node cannot checkpoint its own workload, and the reason generalises

`g7e.2xlarge` has 62 GiB of host RAM and 97,887 MiB of device memory. The driver
copies device memory into host allocations before CRIU sees anything, so the
host needs at least as much free RAM as the GPU has in use. Preflight reports:

```
[blocker] host has 55,956 MiB available but device memory totals 97,887 MiB
```

The vLLM server on this node holds 93,494 MiB of GPU memory. It is not
checkpointable here at any driver version - the arithmetic forbids it.

For spot-instance restoration this is the first thing to check, because it is a
property of the instance type rather than of the software:

| Want to checkpoint | Need |
|---|---|
| a full 98 GiB GPU | a host with >= ~98 GiB free RAM, so ~128 GiB+ of RAM |
| a 40 GiB model | ~64 GiB host RAM |
| this node as configured | reduce `--gpu-memory-utilization`, or a RAM-heavier instance |

The related trap is the spot interruption notice: two minutes. Writing ~90 GiB
of image to disk and then to object storage does not fit in two minutes at any
plausible bandwidth. Spot restoration works when the state is small enough that
the *local* write fits in the notice window and the upload can finish afterwards
- or when the image is written to instance-local NVMe that survives, which it
does not on a reclaimed spot instance. Size the working set accordingly.

## Not measured here

Multi-GPU, NVLS multicast, NCCL, CRIU (the node has none, and Ubuntu 22.04 ships
3.16 against the 4.0 this needs), fabric handles, and driver 610.
