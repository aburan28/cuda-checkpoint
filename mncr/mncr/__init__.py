"""mncr - multi-node checkpoint/restore for CUDA jobs.

Layout mirrors the build plan phases:

    mncr/        shared core: wire protocol, rpc, logging, config   (all phases)
    audit/       allocation auditor and fleet audit                 (P0)
    torchckpt/   in-process rank library                            (P1)
    agent/       privileged per-node agent                          (P2)
    coord/       coordinator, two-phase commit, placement           (P3, P4)
    imagestore/  checkpoint image pipeline                          (P5)
    k8s/         CRDs, controller, admission                        (P6)
    ncclx/       NCCL suspend/resume track                          (P7)
    verify/      correctness, chaos, scale harnesses + fakes        (P8)
"""

from .version import __version__

__all__ = ["__version__"]
