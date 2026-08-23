# Runbook

## Bring-up order

Each step's exit criterion is the next step's precondition. Do not skip ahead;
a failure discovered at step 5 that was visible at step 1 costs a cluster.

1. **`make p0` on every node.** Fix every blocker before anything else. The
   common ones are a driver below 610, a missing `cuda_plugin.so`, and host RAM
   headroom below total device memory. Then `make preflight` on one node, which
   checks the same ground using the real backends and actually creates a job
   file rather than assuming it can.
2. **Run the expandable-segments experiment on one real node.**
   `audit/experiments/expandable_segments.py`. Until it has run you do not know
   whether `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be off
   fleet-wide. Set it to `False` in the meantime.
3. **Label nodes.** `mncr.io/checkpointable=true`, `mncr.io/driver-major`,
   `mncr.io/mnnvl`. The admission webhook and placement both read these.
4. **`make smoke` on one node.** Four levels, in order: driver only, plus
   CRIU, through the agent, then a full epoch. A failure at level *n* makes
   every level above it meaningless, so fix and rerun rather than reading on.
5. **Apply CRDs, RBAC, the DaemonSet, the coordinator, the webhook.** The
   DaemonSet runs preflight as an init container, so a node that cannot
   participate fails at rollout rather than at somebody's commit point.
6. **Add `torchckpt.init()` and a `safe_point()` to the training loop.** Start
   with `strict_clean=True`. A rank that cannot prove it is clean should fail
   loudly at prepare time, when failure is still free.
7. **One `GpuCheckpoint` with `mode: continue` on a single-node job.**
8. **Scale to multi-node, then to a `CheckpointPolicy`.**

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

## What to watch

Two series, and the distinction between them is the whole operational model:

```
mncr_epochs_total{outcome="aborted"}   failed before the commit point
mncr_epochs_total{outcome="failed"}    failed after it
```

An `aborted` rate above zero is a bug to fix at leisure - the job survived every
one of them. A single `failed` means a job was lost. Alert on the second
immediately; trend the first.

Beyond those: `mncr_stopped_seconds` is the number users feel, and
`mncr_gate_findings_total{kind=...}` tells you exactly which resource ranks keep
failing to release, which is usually the fastest route to the cause.

## Retention

`CheckpointPolicy.spec.retain` bounds how many images a job keeps; the
controller sweeps after every successful policy checkpoint, and `mncrctl gc`
does it on demand. Two things are never deleted: the newest `retain` images, and
whatever the job's last-good pointer names — deleting the image a failed epoch
would fall back to is the one mistake retention must not make.

The epoch record survives its image, marked `pruned`. Knowing an image once
existed and was reclaimed is worth a few hundred bytes when somebody asks where
it went.

## When a policy keeps failing

After `suspendAfterFailures` consecutive failures the controller sets
`status.suspended` and stops attempting. Retrying a broken policy turns one
broken job into load on every node it touches. Clear the flag by hand once the
cause is fixed — the suspension is meant to be noticed.

## What to do when the coordinator restarts mid-epoch

`coord.recover()` reports what it found, and never resumes anything by itself.
Epochs that were before the commit point are abortable - unlock the ranks and
retry. Epochs past it are lost; restore the last good image.
