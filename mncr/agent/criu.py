"""CRIU wrapper.

The flags here are the ones that actually matter in a Kubernetes pod, and each
is load-bearing:

  --shell-job          the rank is not session leader of its own session
  --tcp-close          drop TCP rather than repair it. Cross-node peers are at
                       different addresses after a restore, so repairing
                       connections would restore them into the wrong topology;
                       the rank library re-rendezvouses instead
  --external           every mount and device the pod inherits from the host has
                       to be declared, or the dump refuses
  --file-locks         pods hold flocks on shared volumes
  --link-remap         handles files unlinked but still open

Kubelet's own ContainerCheckpoint is deliberately not used: it will happily dump
a container whose CUDA state has not been checkpointed. The agent owns ordering.
"""

import json
import os
import shutil
import subprocess

from mncr import log
from mncr.errors import CriuError

_LOG = log.get("agent.criu")

DEFAULT_EXTERNAL = (
    "mnt[]:m",              # inherited mounts, resolved by the restore side
)


class CriuBackend:
    def __init__(self, binary="criu", libdir="/usr/lib/criu", timeout=3600.0):
        self.binary = binary
        self.libdir = libdir
        self.timeout = timeout

    def _run(self, args, timeout=None):
        cmd = [self.binary] + args
        _LOG.info("criu", cmd=" ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else ""))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout or self.timeout
            )
        except FileNotFoundError as exc:
            raise CriuError(f"{self.binary} not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise CriuError(f"criu timed out after {timeout or self.timeout}s") from exc
        if proc.returncode != 0:
            raise CriuError(
                f"criu {args[0]} failed rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[-800:]}"
            )
        return proc.stdout

    def _common(self, images_dir, external=(), extra=()):
        args = [
            "--images-dir",
            images_dir,
            "--libdir",
            self.libdir,
            "--shell-job",
            "--tcp-close",
            "--file-locks",
            "--link-remap",
        ]
        for item in list(DEFAULT_EXTERNAL) + list(external):
            args += ["--external", item]
        args += list(extra)
        return args

    def pre_dump(self, pid, images_dir, prev_dir=None, external=(), extra=()):
        """Copy pages while the process still runs, to shrink the stop window.

        Only useful before the lock. Once the driver has copied device memory
        into host allocations, a pre-dump would be copying the very pages that
        are about to change.
        """
        os.makedirs(images_dir, exist_ok=True)
        args = ["pre-dump", "--tree", str(pid)] + self._common(
            images_dir, external, extra
        )
        if prev_dir:
            args += ["--prev-images-dir", prev_dir, "--track-mem"]
        self._run(args)
        return images_dir

    def dump(self, pid, images_dir, leave_running=False, external=(), extra=()):
        os.makedirs(images_dir, exist_ok=True)
        args = ["dump", "--tree", str(pid)] + self._common(images_dir, external, extra)
        if leave_running:
            args.append("--leave-running")
        self._run(args)
        return images_dir

    def restore(self, images_dir, detached=True, external=(), extra=(), pidfile=None):
        """Restore and return the pid of the restored process.

        The pid is not optional information: on a node that did not take the
        checkpoint, the agent has no registry entry for this rank, and the pid
        is what every subsequent driver call needs. --pidfile is the only way
        criu reports it back for a detached restore.
        """
        pidfile = pidfile or os.path.join(images_dir, "restored.pid")
        args = ["restore", "--pidfile", pidfile] + self._common(
            images_dir, external, extra
        )
        if detached:
            args.append("--restore-detached")
        self._run(args)
        try:
            with open(pidfile) as fh:
                return int(fh.read().strip())
        except (FileNotFoundError, ValueError) as exc:
            raise CriuError(
                f"criu restore did not write a usable pidfile at {pidfile}"
            ) from exc

    def available(self):
        return shutil.which(self.binary) is not None


class FakeCriuBackend(CriuBackend):
    """Writes a manifest instead of an image. Enough to exercise sequencing."""

    def __init__(self, *_args, **_kwargs):
        self.dumps = []
        self.restores = []
        self._fail = set()

    def fail_next(self, op):
        self._fail.add(op)
        return self

    def _check(self, op):
        if op in self._fail:
            self._fail.discard(op)
            raise CriuError(f"injected criu {op} failure")

    def pre_dump(self, pid, images_dir, prev_dir=None, external=(), extra=()):
        self._check("pre_dump")
        os.makedirs(images_dir, exist_ok=True)
        return images_dir

    def dump(self, pid, images_dir, leave_running=False, external=(), extra=()):
        self._check("dump")
        os.makedirs(images_dir, exist_ok=True)
        with open(os.path.join(images_dir, "fake-image.json"), "w") as fh:
            json.dump({"pid": pid, "leave_running": leave_running}, fh)
        self.dumps.append((pid, images_dir))
        return images_dir

    def restore(self, images_dir, detached=True, external=(), extra=(), pidfile=None):
        self._check("restore")
        marker = os.path.join(images_dir, "fake-image.json")
        if not os.path.exists(marker):
            raise CriuError(f"no image in {images_dir}")
        with open(marker) as fh:
            pid = int(json.load(fh)["pid"])
        self.restores.append(images_dir)
        return pid

    def available(self):
        return True


def make(cfg):
    return (
        FakeCriuBackend()
        if cfg.fake
        else CriuBackend(cfg.criu, cfg.criu_libdir, cfg.dump_timeout)
    )
