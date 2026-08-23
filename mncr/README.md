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

## Try it without a GPU

The simulator runs real agents, a real coordinator and real rank processes
against a fake driver and fake CRIU. The protocol is genuinely under test; only
the hardware is simulated.

```bash
make check     # compile everything, validate manifests, lint shell and C
make test      # 103 unit and protocol tests
make chaos     # fault injection at every phase
make scale     # does wall clock track ranks-per-node or job size?
```

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

Two commands stand between the simulator and real hardware. Run them in order.

```bash
make preflight      # can this node participate? uses the real backends
make smoke          # four levels: driver, +criu, +agent, +full epoch
```

`preflight` checks tooling, driver and CRIU versions, the CUDA plugin,
privileges, host RAM against device memory, and actually creates a job file. It
runs as an init container on the DaemonSet, so a node that cannot participate
never advertises itself as one that can.

`smoke` runs the real driver and real CRIU against a real CUDA process, and
proves the device memory survived by checksum. Level 4 is a full coordinator
epoch — the same simulator, with the fakes swapped out.

## Operating it

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

Everything in `make verify` runs here and passes: the phase model, the
two-phase commit under eight injected faults, device maps, placement, the image
pipeline round trip with checksums, a cross-node restore that has to fetch its
shards back because the images were deleted from the target node, retention and
policy suspension, the admission webhook including its behaviour during an API
outage, and the full rank lifecycle across a simulated cluster.

Nothing has run against a GPU, a real driver, real CRIU, or a real cluster. The
paths that need hardware are the CLI driver backend, the CRIU backend, the
interposer, and the expandable-segments experiment. They are written and
compile; they are not proven. `make preflight` and `make smoke` are what prove
them, and they are the first thing to run on a node.

## Known open question

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocates through
`cuMemCreate`/`cuMemMap`. The documented limitation names the *export*, not the
allocation. Whether the driver rejects a process merely holding VMM allocations
decides a fleet-wide policy, and it is not answerable from documentation. Run
`audit/experiments/expandable_segments.py` on one real node before designing
around either answer. Until then the manifests set it to `False`.
