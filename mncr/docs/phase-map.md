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
| P6 | admission webhook server | `k8s/admission_server.py` | fail-open on API outage, node cache, health probes |
| P6 | retention and policy suspension | `k8s/controller.py`, `coord/main.py` | last-good never deleted; suspend after N failures |
| P6 | container images, CI | `Dockerfile`, `.github/workflows/verify.yml` | two targets, stdlib only; CI runs the whole suite |
| — | operator CLI | `mncrctl` | exercised against a live coordinator |
| — | node preflight | `agent/preflight.py` | real backends; runs as a DaemonSet init container |
| — | on-node smoke test | `verify/smoke.py`, `verify/smoke_target.cu` | four levels; skips cleanly without hardware |
| — | mutating admission | `k8s/admission.py` | injects mount, env and job-file path; never overrides an author |
| — | metrics | `mncr/metrics.py` | Prometheus text; aborted and failed kept as separate series |
| P7 | NCCL seam + benchmark | `ncclx/` | seam refuses the unsafe fast path; patch specified |
| P8 | simulator, chaos, scale | `verify/` | all green |

## Not built

* The NCCL network-suspend patch itself. It needs the NCCL source tree;
  `ncclx/README.md` specifies it.
* TLS certificate issuance for the webhook. The Deployment mounts
  `mncr-admission-tls`; producing it is left to cert-manager or your own CA.
* Object-store credentials handling. `RemoteBackend` shells to whatever CLI the
  cluster already trusts.
