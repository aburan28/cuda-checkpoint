# Plan phase to code

| Phase | Deliverable | Where | Exit criterion status |
|---|---|---|---|
| P0 | allocation auditor | `audit/interpose/cuda_audit.c` | logic tested against a stand-in driver, including the cuGetProcAddress redirect; unproven against real CUDA |
| P0 | fleet audit + report | `audit/fleet_audit.sh`, `audit/report.py` | runs; produces a go/no-go verdict |
| P0 | expandable-segments experiment | `audit/experiments/expandable_segments.py` | runs; skips cleanly without CUDA |
| P1 | rank library | `torchckpt/` | full lifecycle in the simulator; step agreement and rebuild proven with real gloo collectives, and with real NCCL across two nodes; the request itself travels in a per-step collective so ranks enter an epoch together |
| P1 | bind shim for migrated ranks | `torchckpt/netmap/` | NCCL caches the node address at first init; the preload rebinds on the new node. Proven: a job migrated between two nodes, twice |
| P1 | clean-room assertion | `torchckpt/cleanroom.py`, `mncr/procscan.py` | unit tested against fixture procfs |
| P2 | node agent | `agent/main.py` | drives the full sequence in the simulator |
| P2 | driver + CRIU backends | `agent/driver.py`, `agent/criu.py` | proven on hardware; device-mapped restores reach the CRIU CUDA plugin through a `cuda-checkpoint` shim on PATH |
| P2 | node self-description | `agent/nodeinfo.py` | GPU UUIDs, driver, RAM headroom and the node address, served to the coordinator on registration |
| P2 | job files, pid translation | `agent/jobfile.py`, `agent/pids.py` | single-use rule unit tested |
| P3 | two-phase commit | `coord/epoch.py` | eight injected faults, both invariants hold |
| P3 | epoch store | `coord/store.py` | restart classification tested on both sides of the commit point |
| P4 | device maps | `coord/devicemap.py` | partial maps rejected; cross-node restore tested in the simulator and on two real nodes |
| P4 | rendezvous per epoch | `coord/rendezvous.py` | a fresh `tcp://` address on rank 0's current node for every rebuild; follows rank 0 through a migration |
| P4 | placement admission | `coord/placement.py` | every rejection reason unit tested |
| P5 | image pipeline | `imagestore/pipeline.py` | byte-identical round trip, corruption detected |
| P5 | pipeline wired into dump/restore | `agent/main.py` | cross-node fetch tested: images deleted, restore pulls shards back; manifests travel with the shards, found by source node |
| P5 | pre-dump pass | `agent/main.py`, `coord/epoch.py` | runs before the lock; failure is non-fatal |
| P5 | tiered storage + cache | `imagestore/backends.py`, `cache.py` | eviction policy tested |
| P6 | CRDs, controller, admission | `k8s/` | manifests validate; admission unit tested |
| P6 | admission webhook server | `k8s/admission_server.py` | fail-open on API outage, node cache, health probes |
| P6 | retention and policy suspension | `k8s/controller.py`, `coord/main.py` | last-good never deleted; suspend after N failures |
| P6 | container images, CI | `Dockerfile`, `.github/workflows/verify.yml` | two targets, stdlib only; CI runs the whole suite |
| — | operator CLI | `mncrctl` | exercised against a live coordinator |
| — | node preflight | `agent/preflight.py` | real backends; runs as a DaemonSet init container |
| — | on-node smoke test | `verify/smoke.py`, `verify/smoke_target.cu` | six levels; skips cleanly without hardware; level 5 needs two GPUs |
| — | multi-node harness | `verify/cluster.py` | continue, restore, migrate across real nodes; all three pass on two g7e.2xlarge |
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
