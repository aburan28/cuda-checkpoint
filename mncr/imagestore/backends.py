"""Where shards go.

LocalBackend is the node NVMe tier and is always present. RemoteBackend shells
out to whatever CLI the cluster already trusts (aws s3, s5cmd, rclone, gsutil)
rather than binding an SDK - the agent runs privileged in somebody's cluster and
every dependency it carries is a dependency they have to audit.
"""

import os
import shutil
import subprocess

from mncr import log

_LOG = log.get("imagestore.backend")


class Backend:
    def put(self, local_path, key):
        raise NotImplementedError

    def get(self, key, local_path):
        raise NotImplementedError

    def exists(self, key):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key):
        return os.path.join(self.root, key)

    def put(self, local_path, key):
        dest = self._path(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.abspath(local_path) != os.path.abspath(dest):
            shutil.copy2(local_path, dest)
        return dest

    def get(self, key, local_path):
        src = self._path(key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(local_path):
            shutil.copy2(src, local_path)
        return local_path

    def exists(self, key):
        return os.path.exists(self._path(key))

    def delete(self, key):
        try:
            os.unlink(self._path(key))
            return True
        except FileNotFoundError:
            return False


class RemoteBackend(Backend):
    """Generic CLI backend. `template` uses {src} and {dst} placeholders."""

    def __init__(self, prefix, put_template=None, get_template=None, timeout=3600,
                 delete_template=None):
        self.prefix = prefix.rstrip("/")
        self.put_template = put_template or "aws s3 cp {src} {dst}"
        self.get_template = get_template or "aws s3 cp {src} {dst}"
        # Retention has to reach the object store too, or the bytes it
        # reclaims on NVMe live on forever behind it. {dst} is the object.
        self.delete_template = delete_template or "aws s3 rm {dst}"
        self.timeout = timeout

    def _url(self, key):
        return f"{self.prefix}/{key}"

    def _run(self, template, src, dst):
        cmd = template.format(src=src, dst=dst).split()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(cmd[:3])} failed rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:300]}"
            )
        return dst

    def put(self, local_path, key):
        return self._run(self.put_template, local_path, self._url(key))

    def get(self, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return self._run(self.get_template, self._url(key), local_path)

    def exists(self, key):
        return False   # the cache layer treats unknown as a miss

    def delete(self, key):
        self._run(self.delete_template, "", self._url(key))
        return True


class TieredBackend(Backend):
    """Local first, remote behind it. Restores hit NVMe when the image is warm."""

    def __init__(self, local, remote=None):
        self.local = local
        self.remote = remote

    def put(self, local_path, key):
        path = self.local.put(local_path, key)
        if self.remote:
            self.remote.put(path, key)
        return path

    def get(self, key, local_path):
        if self.local.exists(key):
            _LOG.debug("cache hit", key=key)
            return self.local.get(key, local_path)
        if not self.remote:
            raise FileNotFoundError(key)
        _LOG.info("cache miss, fetching", key=key)
        self.remote.get(key, local_path)
        self.local.put(local_path, key)
        return local_path

    def exists(self, key):
        return self.local.exists(key) or (self.remote and self.remote.exists(key))

    def delete(self, key):
        """Both tiers. A remote failure is logged and reported, not raised:
        the sweep must finish, and the epoch stays unpruned to be retried."""
        removed = self.local.delete(key)
        if self.remote:
            try:
                removed = self.remote.delete(key) or removed
            except Exception as exc:  # noqa: BLE001
                _LOG.warn("remote delete failed", key=key, error=str(exc))
                raise
        return removed
