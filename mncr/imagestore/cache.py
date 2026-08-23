"""Node-local warm cache.

The cold-start use case restores the same image over and over, so the useful
policy is not LRU by access but "keep whole images, evict whole images" - half
an image on NVMe is worth nothing, and the restore path cannot start until the
last shard lands.
"""

import os
import shutil
import time

from mncr import log

_LOG = log.get("imagestore.cache")


class ImageCache:
    def __init__(self, root, capacity_bytes=None):
        self.root = root
        self.capacity_bytes = capacity_bytes
        os.makedirs(root, exist_ok=True)

    def _image_dir(self, image_id):
        return os.path.join(self.root, image_id)

    def has(self, image_id):
        return os.path.exists(os.path.join(self._image_dir(image_id), ".complete"))

    def mark_complete(self, image_id):
        path = self._image_dir(image_id)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, ".complete"), "w") as fh:
            fh.write(str(time.time()))
        return path

    def touch(self, image_id):
        marker = os.path.join(self._image_dir(image_id), ".complete")
        if os.path.exists(marker):
            os.utime(marker, None)

    def size_of(self, image_id):
        total = 0
        for dirpath, _dirs, files in os.walk(self._image_dir(image_id)):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except FileNotFoundError:
                    pass
        return total

    def usage(self):
        return sum(self.size_of(name) for name in self.images())

    def images(self):
        try:
            return [
                name
                for name in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, name))
            ]
        except FileNotFoundError:
            return []

    def evict(self, image_id):
        shutil.rmtree(self._image_dir(image_id), ignore_errors=True)
        _LOG.info("evicted", image=image_id)
        return True

    def enforce(self, keep=()):
        """Evict whole images, oldest first, until under capacity."""
        if not self.capacity_bytes:
            return []
        keep = set(keep)
        entries = []
        for image_id in self.images():
            marker = os.path.join(self._image_dir(image_id), ".complete")
            age = os.path.getmtime(marker) if os.path.exists(marker) else 0
            entries.append((age, image_id, self.size_of(image_id)))
        entries.sort()

        total = sum(size for _, _, size in entries)
        evicted = []
        for _age, image_id, size in entries:
            if total <= self.capacity_bytes:
                break
            if image_id in keep:
                continue
            self.evict(image_id)
            evicted.append(image_id)
            total -= size
        return evicted
