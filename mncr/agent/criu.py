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

## Restoring onto different hardware

CRIU's CUDA plugin restores CUDA itself, from inside `criu restore`, by running
`cuda-checkpoint --action restore` while the task is still frozen. It has no
notion of a device map, and CRIU 4.x refuses to restore an image whose
inventory names a plugin that is not loaded - so the plugin can be neither
bypassed nor told about the migration. What it can be given is a different
`cuda-checkpoint`: the plugin resolves the binary through PATH, so the agent
puts a shim ahead of the real one that appends `--device-map` to exactly the
restore call, taken from `MNCR_DEVICE_MAP`. Every other invocation - the `-h`
capability probe, `--get-restore-tid`, `--get-state`, lock, unlock - passes
straight through. Same-hardware restores never set the variable and never see
the shim's branch.
"""

import json
import os
import shlex
import shutil
import subprocess
import tempfile

from mncr import log
from mncr.errors import CriuError

_LOG = log.get("agent.criu")

DEFAULT_EXTERNAL = (
    "mnt[]:m",              # inherited mounts, resolved by the restore side
)

SHIM_NAME = "cuda-checkpoint"
SHIM_TEMPLATE = """#!/bin/sh
# Installed by the mncr agent. See agent/criu.py.
real={real}
if [ -n "${{MNCR_DEVICE_MAP:-}}" ]; then
  action=""; prev=""; mapped=0
  for a in "$@"; do
    [ "$prev" = "--action" ] && action="$a"
    [ "$a" = "--device-map" ] && mapped=1
    prev="$a"
  done
  if [ "$action" = "restore" ] && [ "$mapped" -eq 0 ]; then
    exec "$real" "$@" --device-map "$MNCR_DEVICE_MAP"
  fi
fi
exec "$real" "$@"
"""


class CriuBackend:
    def __init__(self, binary="criu", libdir="/usr/lib/criu", timeout=3600.0,
                 cuda_checkpoint="cuda-checkpoint", shim_dir=None):
        self.binary = binary
        self.libdir = libdir
        self.timeout = timeout
        self.cuda_checkpoint = cuda_checkpoint
        self.shim_dir = shim_dir or os.path.join(tempfile.gettempdir(), "mncr-criu-shim")

    def shim_env(self, device_map):
        """Environment for a criu restore that must apply `device_map`."""
        real = shutil.which(self.cuda_checkpoint)
        if not real:
            raise CriuError(
                f"cannot install the cuda-checkpoint shim: {self.cuda_checkpoint} "
                f"is not on PATH"
            )
        os.makedirs(self.shim_dir, exist_ok=True)
        path = os.path.join(self.shim_dir, SHIM_NAME)
        content = SHIM_TEMPLATE.format(real=shlex.quote(os.path.abspath(real)))
        try:
            with open(path) as fh:
                current = fh.read()
        except OSError:
            current = None
        if current != content:
            tmp = f"{path}.tmp"
            with open(tmp, "w") as fh:
                fh.write(content)
            os.chmod(tmp, 0o755)
            os.replace(tmp, path)
        env = dict(os.environ)
        env["PATH"] = self.shim_dir + os.pathsep + env.get(
            "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        env["MNCR_DEVICE_MAP"] = str(device_map)
        return env

    def _run(self, args, timeout=None, env=None):
        cmd = [self.binary] + args
        _LOG.info("criu", cmd=" ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else ""))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout or self.timeout,
                env=env,
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

    def restore(self, images_dir, detached=True, external=(), extra=(), pidfile=None,
                device_map=None):
        """Restore and return the pid of the restored process.

        The pid is not optional information: on a node that did not take the
        checkpoint, the agent has no registry entry for this rank, and the pid
        is what every subsequent driver call needs. --pidfile is the only way
        criu reports it back for a detached restore.

        device_map, when given, reaches the CUDA plugin through the shim
        described at the top of this module. Pass it only for a restore onto
        hardware other than the image's own; an identity map is noise.
        """
        pidfile = pidfile or os.path.join(images_dir, "restored.pid")
        args = ["restore", "--pidfile", pidfile] + self._common(
            images_dir, external, extra
        )
        if detached:
            args.append("--restore-detached")
        env = None
        if device_map:
            env = self.shim_env(device_map)
            _LOG.info("restoring with a device map via the plugin shim",
                      device_map=device_map, shim=self.shim_dir)
        self._run(args, env=env)
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
        self.restore_maps = []   # device_map given to each restore, None if none
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

    def restore(self, images_dir, detached=True, external=(), extra=(), pidfile=None,
                device_map=None):
        self._check("restore")
        marker = os.path.join(images_dir, "fake-image.json")
        if not os.path.exists(marker):
            raise CriuError(f"no image in {images_dir}")
        with open(marker) as fh:
            pid = int(json.load(fh)["pid"])
        self.restores.append(images_dir)
        self.restore_maps.append(device_map)
        return pid

    def available(self):
        return True


def make(cfg):
    if cfg.fake:
        return FakeCriuBackend()
    return CriuBackend(
        cfg.criu,
        cfg.criu_libdir,
        cfg.dump_timeout,
        cuda_checkpoint=cfg.cuda_checkpoint,
        shim_dir=os.path.join(cfg.cache_dir, "criu-shim"),
    )
