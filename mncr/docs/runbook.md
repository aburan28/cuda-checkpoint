# Runbook

## Provisioning a node

```bash
sudo ./bootstrap-node.sh --cuda-checkpoint /path/to/cuda-checkpoint
```

Idempotent, so it is safe in cloud-init or a DaemonSet init container. Put it
there rather than running it by hand: a spot node that gets reclaimed takes its
hand-built CRIU with it, and the next one comes back bare.

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
4. **`make smoke` on one node.** Six levels, in order: driver only, plus
   CRIU, through the agent, a full epoch, the same with real NCCL (needs two
   GPUs), then checkpoint-stop-restore from images. A failure at level *n*
   makes every level above it meaningless, so fix and rerun rather than
   reading on. Then **`verify/cluster.py` on two nodes**: continue, restore,
   migrate. Nothing multi-node is proven until that table is green.
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
NIXL handle nobody remembered, or one of the two measured on every node so far:

- `gdrcopy_fd(/dev/gdrdrv)`: libfabric (aws-ofi-nccl) opens it at plugin init
  and never closes it, EFA or not. Launch with `FI_HMEM_CUDA_USE_GDRCOPY=0`.
- `listening_socket(127.0.0.1:28028)` and one on the node address: NCCL RAS.
  Launch with `NCCL_RAS_ENABLE=0`. It also saves ten seconds of communicator
  init.

Neither can be released from inside the process; both have to be prevented
at launch. The webhook injects both variables when a pod does not set them.

**Migration fails in NCCL with `Cannot assign requested address`.** The rank
was restored on another node and NCCL is listening on the address it cached
at its first initialisation. Ranks that may be migrated need
`LD_PRELOAD=<control root>/lib/libmncr_netmap.so`; the agent publishes it
there and the webhook injects the variable.

**Restore fails with `bad build-ID` or `bad mode`.** The target node has a
different build of a library the rank mapped, or the same file with a
different mode. CRIU is right to refuse. The usual cause is unattended
upgrades on one node and not another; freeze node images, and run
`verify/cluster.py`, which fingerprints the libraries ranks map and warns when
nodes differ.

**A fresh rank services an epoch nobody asked for.** A request file outlived
a terminal failure. Ranks now ignore requests issued before they started and
the coordinator clears request files on terminal failure; on an older
deployment, delete `<control root>/jobs/<job>/request.json`.

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
