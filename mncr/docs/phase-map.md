# Plan phase to code

| Phase | Deliverable | Where | Exit criterion status |
|---|---|---|---|
| P0 | allocation auditor | `audit/interpose/cuda_audit.c` | compiles; needs a GPU node to run |
| P0 | fleet audit + report | `audit/fleet_audit.sh`, `audit/report.py` | runs; produces a go/no-go verdict |
| P0 | expandable-segments experiment | `audit/experiments/expandable_segments.py` | runs; skips cleanly without CUDA |
| P1 | rank library | `torchckpt/` | full lifecycle exercised in the simulator |
| P1 | clean-room assertion | `torchckpt/cleanroom.py`, `mncr/procscan.py` | unit tested against fixture procfs |
| P2 | node agent | `agent/main.py` | drives the full sequence in the simulator |
| P2 | driver + CRIU backends | `agent/driver.py`, `agent/criu.py` | fakes tested; CLI paths unproven |
| P2 | job files, pid translation | `agent/jobfile.py`, `agent/pids.py` | single-use rule unit tested |
| P3 | two-phase commit | `coord/epoch.py` | eight injected faults, both invariants hold |
| P3 | epoch store | `coord/store.py` | recovery classification tested |
| P4 | device maps | `coord/devicemap.py` | partial maps rejected; cross-node restore tested |
| P4 | placement admission | `coord/placement.py` | every rejection reason unit tested |
| P5 | image pipeline | `imagestore/pipeline.py` | byte-identical round trip, corruption detected |
| P5 | pipeline wired into dump/restore | `agent/main.py` | cross-node fetch tested: images deleted, restore pulls shards back |
| P5 | pre-dump pass | `agent/main.py`, `coord/epoch.py` | runs before the lock; failure is non-fatal |
| P5 | tiered storage + cache | `imagestore/backends.py`, `cache.py` | eviction policy tested |
| P6 | CRDs, controller, admission | `k8s/` | manifests validate; admission unit tested |
| P7 | NCCL seam + benchmark | `ncclx/` | seam refuses the unsafe fast path; patch specified |
| P8 | simulator, chaos, scale | `verify/` | all green |

## Not built

* The NCCL network-suspend patch itself. It needs the NCCL source tree;
  `ncclx/README.md` specifies it.
* A mutating webhook to inject the job-file env var. `k8s/admission.py` has the
  helper; the validating path is what is wired.
* Object-store credentials handling. `RemoteBackend` shells to whatever CLI the
  cluster already trusts.
