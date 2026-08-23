"""Durable epoch state.

The coordinator can be restarted while an epoch is in flight, and what it needs
on restart is not the epoch's progress but its phase - specifically, whether the
commit point was crossed. An epoch found in CHECKPOINTED or later after a
coordinator crash cannot be resumed in place; the job is gone and the last good
image is the recovery path.
"""

import json
import os
import threading
import time


class EpochStore:
    def __init__(self, root="/var/lib/mncr/epochs"):
        self.root = root
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

    def _path(self, epoch_id):
        return os.path.join(self.root, f"{epoch_id}.json")

    def _job_path(self, job_id):
        return os.path.join(self.root, f"job-{job_id}.json")

    def put(self, epoch):
        with self._lock:
            path = self._path(epoch["epoch_id"])
            tmp = f"{path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(epoch, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        return epoch

    def get(self, epoch_id):
        try:
            with open(self._path(epoch_id)) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None

    def list_epochs(self, job_id=None):
        out = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json") or name.startswith("job-"):
                continue
            try:
                with open(os.path.join(self.root, name)) as fh:
                    epoch = json.load(fh)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
            if job_id is None or epoch.get("job_id") == job_id:
                out.append(epoch)
        return out

    def mark_last_good(self, job_id, epoch_id, image_id, requirements=None):
        """Record the image a failed epoch should fall back to."""
        payload = {
            "job_id": job_id,
            "epoch_id": epoch_id,
            "image_id": image_id,
            "requirements": requirements or {},
            "at": time.time(),
        }
        with self._lock:
            path = self._job_path(job_id)
            tmp = f"{path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        return payload

    def last_good(self, job_id):
        try:
            with open(self._job_path(job_id)) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None

    def in_flight(self):
        """Epochs that were not finished. Split by whether they committed."""
        recoverable, lost = [], []
        for epoch in self.list_epochs():
            phase = epoch.get("phase")
            if phase in ("running", "aborted", "failed", "resumed"):
                continue
            if phase in ("preparing", "prepared", "locked"):
                recoverable.append(epoch)
            else:
                lost.append(epoch)
        return recoverable, lost
