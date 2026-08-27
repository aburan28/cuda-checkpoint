# Measured across two nodes: driver 595.91.07, NCCL 2.29.7, CRIU 4.2.1

Run 2026-08-26 on two g7e.2xlarge nodes (one RTX PRO 6000 Blackwell each,
62 GiB host RAM, no EFA, sockets between them), torch 2.13.0+cu130, in one
AWS availability zone. Everything below is observed, not inferred; the harness
is `verify/cluster.py` and it prints what it measured.

The single-node findings in [findings-595-blackwell.md](findings-595-blackwell.md)
all held. This document is only what needed two nodes to see.

## The result

```
SCENARIO    RESULT   SECONDS
----------------------------
continue    pass        6.22    checkpoint, keep running; NCCL torn down and rebuilt
restore     pass        7.19    checkpoint and stop; both ranks die; criu brings them back
migrate     pass       16.15    every rank restored on the other node; then checkpointed again
```

- The coordinator fanned out to two agents on two nodes; every phase ran in
  parallel across nodes.
- The ranks were in a real NCCL process group across the network. Before every
  lock the group was destroyed; after every restore it was rebuilt on an
  address the coordinator issued, and every rank proved it with an all_reduce
  over the new group.
- `restore`: both processes were killed by `criu dump` and restored by
  `criu restore`, with their original pids, on their original nodes.
- `migrate`: rank 0 moved from node A to node B and rank 1 from B to A. Each
  target fetched the image through the store, found its manifest by the
  source node's name, restored with a non-identity device map through the
  CRIU CUDA plugin, and resumed on a GPU whose UUID the process had never
  seen. A further checkpoint-and-continue of the migrated job then succeeded.

Seven things stood between the code as it was and that table. Each was
found by the run, not by reasoning, and each is now either fixed or fenced.

## 1. The request file lands on different nodes at different steps

Every node's agent writes `request.json` when the coordinator's `prepare`
reaches it. With gloo on one node the ranks always saw it in the same step.
Across two nodes they need not, and NCCL matches collectives by order: a
rank one step ahead issues its step-agreement all_reduce against its peer's
training all_reduce.

The fix is a control collective at every safe point while a group is live -
`MAX` over `[epoch, lookahead]` - so every rank learns in the same step that
some rank has a request, and what it is. A rank whose file has not arrived
waits `MNCR_REQUEST_SETTLE` (5 s) for it and otherwise acts on what the
collective carried. The cost is one tiny all_reduce and, on NCCL, a device
sync per step.

Seen in the wild before the fix: a request file left behind by a failed epoch
(nothing clears it after a terminal failure) was picked up by a freshly
launched rank on one node and not the other. Two changes follow: a rank
ignores requests issued before it started, and the coordinator clears request
files when an epoch fails past the commit point.

## 2. libfabric holds `/dev/gdrdrv`, and nothing can let go of it

The first two-node epoch aborted: both ranks voted dirty, holding a GDRCopy
fd after `destroy_process_group()`. NCCL's own GDRCopy support is off by
default (`NCCL_GDRCOPY_ENABLE=0` in 2.29.7 and the fd appears with it set).
The owner is the aws-ofi-nccl plugin: libfabric opens `/dev/gdrdrv` during
plugin initialisation, the plugin then finds no EFA device and fails, NCCL
falls back to sockets - and the fd stays open for the life of the process.

CRIU cannot dump it. Destroying the communicator does not close it. It has to
be prevented at launch: `FI_HMEM_CUDA_USE_GDRCOPY=0` (keeps EFA usable) or
`NCCL_NET_PLUGIN=none`. The gate's message now says so; the admission
webhook injects the former; `verify/cluster.py` sets it by default.

The abort itself is worth noting: the gate fired before the lock, the epoch
was abandoned, and the job survived. That is the design working.

## 3. NCCL RAS leaves two listeners per process

After teardown each rank still held two listening sockets: `127.0.0.1:28028`
and one on the node address. They belong to NCCL's RAS subsystem (on by
default since 2.24), are per process rather than per communicator, and are
never closed. Torch's own TCPStore, by contrast, is fully released by
`destroy_process_group()` plus a `gc.collect()`.

CRIU dumps a listening socket fine and on restore binds its port again. The
`restore` scenario lost an epoch to `inet: Can't bind inet socket: Address
already in use` on the same node, past the commit point.

`NCCL_RAS_ENABLE=0` leaves zero sockets after teardown. It also cut
communicator initialisation from 10.5 s to 0.3 s here (RAS spends ten
seconds in bootstrap on this setup). The process scanner now treats any TCP
socket as a before-lock finding, so a RAS-enabled rank aborts cheaply instead
of failing terminally; the webhook injects the variable.

## 4. NCCL remembers the node it was born on

With everything above in place, migration failed in the NCCL warm-up on the
target node: `Call to bind failed: Cannot assign requested address`. NCCL
resolves the node's address once, at its first initialisation, and keeps it
in static memory (the bootstrap interface, the socket transport's device
list). A migrated process carries the old node's address and tries to listen
on it.

There is no API to reset that state and no way to reload libnccl from under
torch. But NCCL advertises every listener from `getsockname()` after
`bind()`, never from its cache. `torchckpt/netmap/libmncr_netmap.so` is a
preload that retries a `bind()` failing with `EADDRNOTAVAIL` on a unicast
IPv4 address using the address this host would route through - the same
choice NCCL makes on a fresh node. Peers learn the new address from the
handshake. Nothing else is touched; a rank that never moves never takes the
branch. With it preloaded, the migrated job rebuilt its group and
checkpointed again.

Anything else that caches the node address in process memory will need the
same treatment. IB verbs would: addresses there are GIDs, not sockets, and
this shim does not reach them.

## 5. Manifests were only ever local

Each node writes its manifest to its own image directory. A node restoring a
rank it never dumped had no manifest to read, and the simulator never
noticed because all of its "nodes" share one disk. Manifests now go into the
store beside the shards, and a restore is told which node dumped each rank so
the target can fetch the right one.

## 6. Nodes drift, and CRIU notices

Two nodes from one AMI, launched minutes apart, refused to migrate between
each other an hour later: `libcrypto.so.3 has bad build-ID`. Unattended
upgrades had run on one of them. CRIU validates the build-id of every mapped
file and is right to; the process would otherwise run a different library at
its saved addresses. It also checks file modes, which caught the shim built
under two different umasks.

The harness fingerprints the libraries every rank maps and warns when nodes
differ. Operationally: image nodes identically and freeze them - container
images do this by construction - and do not let anything upgrade a
checkpointable node in place.

## 7. The CRIU plugin restores CUDA itself, without a device map

Known from single-node work; it is what makes migration possible or not.
`criu restore` calls `cuda-checkpoint --action restore` from inside the
plugin, with no map, and CRIU 4.x refuses to restore an image whose inventory
names a plugin that is not loaded. The plugin resolves the binary through
`PATH`, so the agent puts a shim ahead of it for device-mapped restores that
appends `--device-map` to exactly that call. Both migrated ranks restored this
way, onto GPUs with UUIDs their images had never seen.

## Timing

| Operation | Time |
|---|---|
| checkpoint-and-continue, 2 ranks, 2 nodes | 5.0-5.3 s |
| checkpoint-and-stop | 4.5 s |
| restore in place from images | ~2.7 s after the stop |
| migrate (fetch, restore with map, rebuild) | ~7 s after the stop |
| NCCL communicator init, RAS on / off | 10.5 s / 0.3 s |

The images were small (a 1M-element tensor per rank); the numbers measure the
protocol and the tooling, not bandwidth.

## Not measured here

- **Multiple GPUs in one node.** The account's G-instance quotas allowed no
  4-GPU node, so intra-node NCCL - P2P through imported `cuMem` handles, the
  case the single-node findings showed to be unrestorable if left mapped -
  is still unmeasured. `make smoke` level 5 runs it on any node with two or
  more GPUs.
- **IB/RoCE and EFA.** Sockets only. Whether the OFI plugin releases its
  verbs fds on communicator destroy, and how a migrated rank's GIDs are
  handled, are open.
- **Driver 610, NVLS, fabric handles.** Unchanged from the single-node list.
- **Large images.** Bandwidth through the store was not the subject.

## Reproducing

Two nodes, identical, with the repo at the same path, a python that has torch
on both, root able to ssh to the other as an unprivileged user, and an rsync
destination for the store:

```bash
sudo ./bootstrap-node.sh --cuda-checkpoint ../bin/x86_64_Linux/cuda-checkpoint   # each node
sudo python3 -m verify.cluster \
    --node node-a=172.31.6.202 --node node-b=172.31.4.217 \
    --store ubuntu@172.31.6.202:/var/lib/mncr/store \
    --python /opt/torch/bin/python --repo /home/ubuntu/cuda-checkpoint/mncr
```
