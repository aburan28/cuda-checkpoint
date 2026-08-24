# mncr - multi-node checkpoint/restore for CUDA jobs on Kubernetes

An implementation of the build plan: end-to-end checkpoint and restore of
multi-node PyTorch jobs, built on `cuda-checkpoint` and CRIU.

The premise is that everything the driver cannot checkpoint can be destroyed
before the checkpoint and rebuilt after it. That is what makes this buildable
today without waiting on a driver release - and it is also what you give up:
transparency. The application has to be checkpoint-aware.

**Scope.** Kubernetes, HGX nodes with IB/RoCE, single-node NVLS, PyTorch + NCCL.
MNNVL/NVL72 is explicitly out of scope and refused by admission, because fabric
handles cannot be checkpointed and there is no workaround.

## The one thing to understand first

There is a commit point, and it sits between `lock` and `checkpoint`.

```
running -> prepare+vote -> lock  |  checkpoint -> dump -> restore+resume
           <----- abortable ---->|<------- no way back -------->
```

Before it, a failure costs one drained step: unlock, resume, retry. After it,
the ranks have released their GPU resources and the driver offers no rollback,
so the epoch is lost and the job falls back to its last good image.

Every design decision here follows from that asymmetry - most of all the rule
that **every rank must vote before a single checkpoint call is issued**.

## Layout

| Path | Phase | What it is |
|---|---|---|
| `mncr/` | all | phase model, RPC, process scanner, config, logging |
| `audit/` | P0 | `LD_PRELOAD` allocation auditor, fleet audit, report, the expandable-segments experiment |
| `torchckpt/` | P1 | in-process rank library: safe points, teardown, `assert_clean()`, rebuild |
| `agent/` | P2 | privileged node agent: driver calls, CRIU, job files, pid translation, verification |
| `coord/` | P3, P4 | coordinator, two-phase commit, device maps, placement |
| `imagestore/` | P5 | shard, compress, checksum, tiered storage, warm cache |
| `k8s/` | P6 | CRDs, controller, admission, manifests |
| `ncclx/` | P7 | NCCL strategy seam, benchmark, patch plan |
| `verify/` | P8 | cluster simulator, chaos matrix, scale ladder, on-node smoke test |
| `tests/` | P8 | unit and protocol tests |
| `mncrctl` | — | operator CLI |
| `mncr/metrics.py` | — | Prometheus endpoint on the agent and coordinator |

## Try it without a GPU

The simulator runs real agents, a real coordinator and real rank processes
against a fake driver and fake CRIU. The protocol is genuinely under test; only
the hardware is simulated.

```bash
make check     # compile everything, validate manifests, lint shell and C
make test      # 136 unit and protocol tests
make chaos     # fault injection at every phase
make scale     # does wall clock track ranks-per-node or job size?
```

Two of those deserve calling out, because they cover the parts that are
otherwise only testable on hardware:

- **The interposer runs against a stand-in driver.** `make check` builds
  `cuda_audit.so`, loads it over a fake libcuda and asserts what it recorded —
  including that a call resolved through `cuGetProcAddress` went through the
  wrapper. Interposing the symbol alone would pass a naive test and silently
  miss every allocation NCCL makes.
- **Step agreement runs against a real process group.** gloo on CPU gives real
  collectives without a GPU, so `tests/test_collectives.py` proves ranks
  arriving at different local steps all stop at the same one, and that a torn
  down group comes back working — after a successful epoch and after an abort.

`make chaos` asserts the two invariants that matter:

```
CASE                 SIDE           RAISED     PHASE      RESULT
dirty-rank           before-commit  abortable  aborted    pass
lock-failure         before-commit  abortable  aborted    pass
rank-killed          before-commit  abortable  aborted    pass
agent-unreachable    before-commit  abortable  aborted    pass
checkpoint-failure   after-commit   terminal   failed     pass
dump-failure         after-commit   terminal   failed     pass
resume-failure       after-commit   terminal   failed     pass
unlock-failure       after-commit   terminal   failed     pass
```

## Using it from a training loop

```python
import torchckpt

torchckpt.init(job_id="train-7", rank=rank, world_size=world)

@torchckpt.on_quiesce
def teardown():
    torch.cuda.synchronize()
    dist.destroy_process_group()   # frees cuMem allocs, NVLS groups, verbs fds
    kv_transfer.close()            # any other verbs user: GDS, NIXL, UCX
    torch.cuda.empty_cache()

@torchckpt.on_resume
def rebuild(ctx):
    dist.init_process_group("nccl", init_method=ctx.init_method,
                            rank=ctx.rank, world_size=ctx.world_size)

for step in range(steps):
    with torchckpt.safe_point():   # a checkpoint may only be taken here
        train_step()
```

`torchckpt` runs the default PyTorch teardown and rebuild for you; the hooks are
for everything else the process holds. `assert_clean()` runs automatically
before the rank votes, and the agent re-runs the same scan from outside before
it touches the driver.

### Why a safe point, and why ranks agree on a step

Ranks are coupled by collectives but not aligned: when a request lands, one rank
may have finished step 100 while another is midway through it. Restoring a job
whose ranks are on different steps is a correctness bug. So before tearing
anything down, the ranks use the communicator that is about to be destroyed to
agree on a step to stop at, and keep training until they all reach it.

## On a GPU node

```bash
sudo ./bootstrap-node.sh --cuda-checkpoint ../bin/x86_64_Linux/cuda-checkpoint
make preflight      # can this node participate? uses the real backends
make smoke          # four levels: driver, +criu, +agent, +full epoch
```

`bootstrap-node.sh` installs the utility, builds CRIU with its CUDA plugin (no
distro ships 4.x with it, and the plugin is the whole point), and ends by
running preflight. It is idempotent, so it belongs in cloud-init or a DaemonSet
init container — which matters more than it sounds: a spot node reclaimed
mid-session takes a hand-built CRIU with it.

`preflight` checks tooling, driver and CRIU versions, the CUDA plugin,
privileges, host RAM against device memory, and actually creates a job file. It
runs as an init container on the DaemonSet, so a node that cannot participate
never advertises itself as one that can.

`smoke` runs the real driver and real CRIU against a real CUDA process, and
proves the device memory survived by checksum. Level 4 is a full coordinator
epoch — the same simulator, with the fakes swapped out.

## Operating it

Pods labelled `mncr.io/checkpointable=true` are mutated on admission: the
control-directory mount, the `MNCR_*` environment and the job-file path are all
injected, so a rank pod needs the label and nothing else. Anything the author
set explicitly is left alone.

```bash
kubectl apply -f k8s/crds/
kubectl apply -f k8s/manifests/rbac.yaml
kubectl apply -f k8s/manifests/agent-daemonset.yaml
kubectl apply -f k8s/manifests/coordinator.yaml
kubectl apply -f k8s/manifests/admission.yaml   # needs a TLS secret
kubectl apply -f k8s/manifests/webhook.yaml
```

Images: `./prepare-image-context.sh && make images`. Both are stdlib-only
Python with no pip dependency tree — deliberate, because the agent runs
privileged in somebody else's cluster.

```bash
mncrctl status                       # nodes, jobs, recent epochs
mncrctl checkpoint train-7           # or --mode stop, for preemption
mncrctl epochs train-7
mncrctl plan-restore ep-abc --nodes 4    # which nodes could host it, and why not
mncrctl restore train-7 ep-abc --targets node-a=0,1 --targets node-b=2,3
mncrctl gc train-7 --retain 3
mncrctl preflight
```

Metrics are on `:9180` (agent) and `:9181` (coordinator). The series worth
alerting on is the one that separates the two kinds of failure:

```
mncr_epochs_total{outcome="aborted"}   the job survived; something to fix
mncr_epochs_total{outcome="failed"}    the job did not; restore an image
mncr_stopped_seconds_bucket            how long the job was not running
mncr_gate_findings_total{kind=...}     what ranks are still holding
```

```yaml
apiVersion: mncr.io/v1alpha1
kind: GpuCheckpoint
metadata: {name: train-7-now}
spec:
  jobId: train-7
  mode: continue        # or "stop", for preemption
```

The field to read on failure is `status.jobIntact`. See [docs/runbook.md](docs/runbook.md).

## What has and has not been exercised

Proven by test here:

- the phase model and two-phase commit, under eight injected faults
- device maps, placement, retention, policy suspension
- the image pipeline round trip with checksums, and a cross-node restore that
  has to fetch its shards back because the images were deleted from the target
- the admission webhook, including its behaviour during an API outage
- the interposer's recording, severity classification and `cuGetProcAddress`
  redirect, against a stand-in driver
- step agreement, teardown and communicator rebuild, against real gloo
  collectives
- the full rank lifecycle across a simulated cluster

Proven on real hardware — driver 595.91.07, RTX PRO 6000 Blackwell, CRIU 4.2.1:

- all four `make smoke` levels: driver, +CRIU, +agent, +full coordinator epoch,
  with device memory verified by checksum at each
- the `cuda-checkpoint` CLI backend and the CRIU backend
- the interposer under the production `LD_PRELOAD` path
- `make preflight` against a node that really does fail one of its checks

Still not proven:

- NCCL-specific teardown — the gloo tests prove the shape, not the NVLS and
  verbs releases only NCCL performs
- multi-GPU, NVLS multicast, fabric handles, driver 610
- restore onto a *different* node (this was one machine)

See [docs/findings-595-blackwell.md](docs/findings-595-blackwell.md), which
includes the three production bugs that only hardware exposed.

## Measured on hardware

The plan's biggest open question is answered. Full results in
[docs/findings-595-blackwell.md](docs/findings-595-blackwell.md); the two that
change decisions:

**Expandable segments are fine.** A process merely holding `cuMemCreate` /
`cuMemMap` allocations checkpoints and restores cleanly on driver 595. The
manifests no longer force `expandable_segments:False`.

**Exporting is fine; importing is fatal.** The vendor documentation says the
utility "does not support ... IPC memory created with
`cuMemExportToShareableHandle()`". Measured, that splits: the *exporting*
process checkpoints and restores fine even while a peer maps the memory, while
the *importing* process checkpoints and then fails to restore with `"invalid
argument"` — past the commit point, and unrecoverable afterwards. Tested in both
restore orders, so it is the sharing, not the documented ordering rule.

NCCL ranks import each other's handles, so communicator teardown before the lock
is **required**, not merely tidy. And the failure landing after the commit point
is this system's central asymmetry, observed rather than argued.

Caveats: driver 595 on a single Blackwell GPU. 610 and multi-GPU are unmeasured.
