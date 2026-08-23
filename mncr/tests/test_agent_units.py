"""Agent-local rules: job files, driver ordering, verification gates."""

import os
import shutil
import tempfile
import unittest

from . import context  # noqa: F401
from agent.driver import STATE_CHECKPOINTED, STATE_RUNNING, FakeBackend
from agent.jobfile import ENV_VAR, JobFiles
from agent.verify import Verifier
from mncr.errors import DriverError, NotCleanError, PreconditionError
from .test_procscan import build_proc


class TestJobFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.jf = JobFiles(os.path.join(self.tmp, "jobs"), fake=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_then_env(self):
        path = self.jf.create("job-a")
        self.assertTrue(os.path.exists(path))
        self.assertEqual(self.jf.env_for("job-a"), {ENV_VAR: path})

    def test_reuse_refused(self):
        self.jf.create("job-a")
        with self.assertRaises(PreconditionError) as ctx:
            self.jf.create("job-a")
        self.assertIn("single-use", str(ctx.exception))

    def test_env_without_file_refused(self):
        with self.assertRaises(PreconditionError):
            self.jf.env_for("never-created")

    def test_remove_allows_relaunch(self):
        self.jf.create("job-a")
        self.assertTrue(self.jf.remove("job-a"))
        self.assertFalse(self.jf.exists("job-a"))
        self.jf.create("job-a")   # must not raise


class TestDriverOrdering(unittest.TestCase):
    def test_checkpoint_requires_lock(self):
        drv = FakeBackend().add_pid(1)
        with self.assertRaises(DriverError) as ctx:
            drv.checkpoint(1)
        self.assertIn("illegal driver transition", str(ctx.exception))

    def test_double_unlock_rejected(self):
        drv = FakeBackend().add_pid(1)
        drv.lock(1, 100)
        drv.unlock(1)
        with self.assertRaises(DriverError):
            drv.unlock(1)

    def test_restore_requires_checkpointed(self):
        drv = FakeBackend().add_pid(1)
        drv.lock(1, 100)
        with self.assertRaises(DriverError):
            drv.restore(1)

    def test_full_cycle_returns_to_running(self):
        drv = FakeBackend().add_pid(1)
        drv.lock(1, 100)
        drv.checkpoint(1)
        self.assertEqual(drv.state_of(1), STATE_CHECKPOINTED)
        drv.restore(1, device_map="GPU-a=GPU-b")
        drv.unlock(1)
        self.assertEqual(drv.state_of(1), STATE_RUNNING)

    def test_unknown_pid_rejected(self):
        with self.assertRaises(DriverError):
            FakeBackend().lock(999, 100)

    def test_fault_injection_targets_one_pid(self):
        drv = FakeBackend().add_pid(1).add_pid(2)
        drv.fail_next("lock", 2, "boom")
        drv.lock(1, 100)
        with self.assertRaises(DriverError):
            drv.lock(2, 100)


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_before_lock_rejects_verbs_fd(self):
        root = build_proc(self.tmp, 10, fds=["/dev/infiniband/uverbs0"])
        with self.assertRaises(NotCleanError):
            Verifier(root, strict=True).before_lock(10)

    def test_before_lock_allows_device_fd(self):
        root = build_proc(self.tmp, 11, fds=["/dev/nvidia0"])
        self.assertEqual(Verifier(root, strict=True).before_lock(11), [])

    def test_before_dump_rejects_device_fd(self):
        root = build_proc(self.tmp, 12, fds=["/dev/nvidia0"])
        with self.assertRaises(NotCleanError):
            Verifier(root, strict=True).before_dump(12)

    def test_non_strict_reports_without_raising(self):
        root = build_proc(self.tmp, 13, fds=["/dev/infiniband/uverbs0"])
        findings = Verifier(root, strict=False).before_lock(13)
        self.assertEqual(len(findings), 1)


if __name__ == "__main__":
    unittest.main()
