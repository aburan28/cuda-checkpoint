# P7 - the NCCL suspend/resume track

## What is actually missing

NCCL 2.29.7 ships `ncclCommSuspend` / `ncclCommResume`, and the roadmap
([NCCL #2090](https://github.com/nvidia/nccl/issues/2090)) describes them as
*dynamic memory offload*, not checkpoint support. The API defines one flag:

```
NCCL_SUSPEND_MEM  0x01   release dynamic GPU memory allocations
```

That releases the allocations. It does not release:

* **NVLS multicast state** - the failure in
  [NCCL #2337](https://github.com/nvidia/nccl/issues/2337): suspend succeeds,
  then `libcuda` errors on restore because multicast objects are still bound.
* **Network resources** - verbs contexts, queue pairs, GDR registrations. A
  suspended communicator still holds `/dev/infiniband/uverbs*` file
  descriptors, and CRIU cannot dump a process that holds them.

Both are exactly what the checkpoint path cannot handle. So suspend as shipped
is not sufficient, and `seam.py` refuses to select it unless the runtime
advertises flags covering the rest.

## Why we carry a seam rather than just using destroy/rebuild

Destroy-and-rebuild is correct everywhere and is the default here. It costs a
full communicator re-initialisation on resume - roughly 1-3 s at TP=4/8. For
fault tolerance that is noise against the image write. For the cold-start use
case it is most of the budget, which is the entire reason to want the fast path.

`bench.py` turns that into a number for your topology. Run it before deciding
whether to carry an out-of-tree NCCL at all.

## The patch, specifically

Two new flags, and the state each must release:

```
NCCL_SUSPEND_NET   0x02
    destroy verbs QPs, CQs, PDs and MRs; deregister GDR buffers; close the
    uverbs fds. Keep enough peer addressing to rebuild on resume, in host
    memory, since host memory survives the checkpoint.

NCCL_SUSPEND_NVLS  0x04
    unbind multicast objects (cuMulticastUnbind), release the multicast handle,
    and drop the UC mappings backing it. Record the team membership so resume
    can recreate the group with the same participants.
```

`ncclCommResume` mirrors both: re-establish connections from the retained
addressing, then re-create and re-bind the multicast group.

The prototype attached to #2337 covers the NVLS half. The network half is the
part that has to be written.

## Sequencing

1. Vendor NCCL at the version you run, apply the #2337 prototype, add the
   network flag.
2. Export `MNCR_NCCL_SUSPEND_FLAGS=0x7` from the build so `seam.select()` can
   see that the fast path is safe. Without that variable the seam assumes only
   the upstream MEM flag and falls back - deliberately, because a strategy that
   looks cheaper and silently produces unrestorable processes is worse than a
   slow one.
3. Run `bench.py` at your TP width. Carry the patch only if the delta justifies
   an out-of-tree build.
4. File the network half upstream against #2337.

## Status

Not on the funded roadmap. "CUDA checkpoint/restore" sits in the *features under
consideration* bucket of #2090, not in 2.30. Plan on carrying this yourself if
you need it.
