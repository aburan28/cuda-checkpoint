# Runbook

## Bring-up order

Each step's exit criterion is the next step's precondition. Do not skip ahead;
a failure discovered at step 5 that was visible at step 1 costs a cluster.

1. **`make p0` on every node.** Fix every blocker before anything else. The
   common ones are a driver below 610, a missing `cuda_plugin.so`, and host RAM
   headroom below total device memory.
2. **Run the expandable-segments experiment on one real node.**
   `audit/experiments/expandable_segments.py`. Until it has run you do not know
   whether `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be off
   fleet-wide. Set it to `False` in the meantime.
3. **Label nodes.** `mncr.io/checkpointable=true`, `mncr.io/driver-major`,
   `mncr.io/mnnvl`. The admission webhook and placement both read these.
4. **Apply CRDs, RBAC, the DaemonSet, the coordinator.**
5. **Add `torchckpt.init()` and a `safe_point()` to the training loop.** Start
   with `strict_clean=True`. A rank that cannot prove it is clean should fail
   loudly at prepare time, when failure is still free.
6. **One `GpuCheckpoint` with `mode: continue` on a single-node job.**
7. **Scale to multi-node, then to a `CheckpointPolicy`.**

## Reading a failure

The one field that matters on a failed `GpuCheckpoint` is `status.jobIntact`.

| jobIntact | reason | meaning | what to do |
|---|---|---|---|
| `true` | `Aborted` | Failed before the commit point. Every rank is alive and unlocked; the job lost one drained step. | Fix the cause and retry. Nothing is lost. |
| `false` | `LostPastCommitPoint` | Failed after the driver checkpoint. Those ranks released their GPU resources and cannot be resumed in place. | Restore the last good image. The job as it stood is gone. |

`status.message` names the node and the failing call.

## Common causes, in the order they actually occur

**Ranks vote dirty.** `assert_clean()` found something. The finding names the
fd or mapping. Almost always a communicator that was not destroyed, a GDS or
NIXL handle nobody remembered, or `expandable_segments` still on.

**Lock times out.** A rank is still draining work. If it recurs, the lock
timeout is below your longest kernel - raise `MNCR_LOCK_TIMEOUT_MS`. If it
recurs *only* under collectives, a rank is blocked in an allreduce waiting for a
peer, which means prepare did not destroy communicators before the lock. That is
a bug, not a tuning problem.

**Restore fails to place.** `plan_restore` lists every rejected node with a
reason. Driver major mismatch and host RAM headroom are the usual two.

**Restore fails on an MNNVL node.** It should never have scheduled there.
Check the node label and the webhook.

## Operational limits worth knowing before you hit them

* A checkpoint image is about the size of device memory in use. 8xH100 is
  ~640 GiB per node, per epoch. Size `MNCR_IMAGE_DIR` and the object store
  accordingly, and keep the retention policy honest.
* Host RAM must exceed device memory in use. The driver copies device memory
  into host allocations before CRIU ever runs.
* Lock and checkpoint run one process at a time within a node. Wall clock grows
  with ranks-per-node and not with job size; `make scale` demonstrates the
  property.
* Job files are single-use and node-local. Delete a job before relaunching it.
* Restoring across driver major versions is not guaranteed and is refused.

## What to do when the coordinator restarts mid-epoch

`coord.recover()` reports what it found, and never resumes anything by itself.
Epochs that were before the commit point are abortable - unlock the ranks and
retry. Epochs past it are lost; restore the last good image.
