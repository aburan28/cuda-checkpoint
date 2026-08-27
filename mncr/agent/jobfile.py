"""Job file lifecycle for driver 610 IPC.

Two rules from the vendor documentation, both enforced here rather than
documented and hoped for:

    "Job files should never be reused"
    "should only be used in the environment where they were created"

So: one file per job launch, created on node-local tmpfs, refused if it already
exists, and unlinked when the job goes away. A job file that travels between
nodes or outlives its launch is a corruption source, not a convenience.

Creation uses the pattern from the vendor's r610 demo - launch a trivial job and
copy the file the utility hands it, so the ranks do not have to be children of
cuda-checkpoint.
"""

import os
import subprocess

from mncr import log
from mncr.errors import DriverError, PreconditionError

_LOG = log.get("agent.jobfile")

ENV_VAR = "CUDA_CHECKPOINT_JOB_FILE"


class JobFiles:
    def __init__(self, root="/run/mncr/jobs", binary="cuda-checkpoint", fake=False):
        self.root = root
        self.binary = binary
        self.fake = fake

    def path_for(self, job_id):
        return os.path.join(self.root, f"{job_id}.jobfile")

    def create(self, job_id):
        """Create the job file for `job_id`. Refuses to overwrite."""
        dest = self.path_for(job_id)
        if os.path.exists(dest):
            raise PreconditionError(
                f"job file for {job_id} already exists at {dest}; job files are "
                f"single-use - delete the job before relaunching it"
            )
        os.makedirs(self.root, exist_ok=True)

        if self.fake:
            with open(dest, "w") as fh:
                fh.write(f"fake job file for {job_id}\n")
            _LOG.info("job file created (fake)", job=job_id, path=dest)
            return dest

        cmd = [
            self.binary,
            "--launch-job",
            "bash",
            "-c",
            f'cp "${ENV_VAR}" "{dest}"',
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError as exc:
            raise DriverError(f"{self.binary} not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise DriverError("job file creation timed out") from exc
        if proc.returncode != 0 or not os.path.exists(dest):
            raise DriverError(
                f"job file creation failed rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:400]}"
            )
        os.chmod(dest, 0o600)
        _LOG.info("job file created", job=job_id, path=dest)
        return dest

    def env_for(self, job_id):
        """Environment additions for a rank container in this job."""
        path = self.path_for(job_id)
        if not os.path.exists(path):
            raise PreconditionError(f"no job file for {job_id}; create it before launch")
        return {ENV_VAR: path}

    def remove(self, job_id):
        try:
            os.unlink(self.path_for(job_id))
            _LOG.info("job file removed", job=job_id)
            return True
        except FileNotFoundError:
            return False

    def exists(self, job_id):
        return os.path.exists(self.path_for(job_id))
