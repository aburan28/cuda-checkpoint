"""Agent-local rules: job files, driver ordering, verification gates."""

import os
import shutil
import subprocess
import tempfile
import unittest

from . import context  # noqa: F401
from agent.criu import CriuBackend
from agent.driver import STATE_CHECKPOINTED, STATE_RUNNING, FakeBackend
from agent.jobfile import ENV_VAR, JobFiles
from agent.verify import Verifier
from mncr.errors import DriverError, NotCleanError, PreconditionError
from .test_procscan import build_proc


class TestCriuShim(unittest.TestCase):
    """The CUDA plugin restores through whatever `cuda-checkpoint` PATH finds.
    The shim is how a device map reaches it, and it must touch nothing else."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.real = os.path.join(self.tmp, "real-cuda-checkpoint")
        with open(self.real, "w") as fh:
            fh.write('#!/bin/sh\necho "$@"\n')
        os.chmod(self.real, 0o755)
        self.backend = CriuBackend(
            cuda_checkpoint=self.real, shim_dir=os.path.join(self.tmp, "shim")
        )
        self.env = self.backend.shim_env("GPU-a=GPU-b,GPU-c=GPU-d")
        self.shim = os.path.join(self.backend.shim_dir, "cuda-checkpoint")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *args, env=None):
        proc = subprocess.run(
            [self.shim, *args], env=env or self.env, capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip()

    def test_shim_is_first_on_path(self):
        self.assertTrue(os.access(self.shim, os.X_OK))
        self.assertEqual(shutil.which("cuda-checkpoint", path=self.env["PATH"]), self.shim)

    def test_restore_gets_the_map(self):
        self.assertEqual(
            self._run("--action", "restore", "--pid", "7"),
            "--action restore --pid 7 --device-map GPU-a=GPU-b,GPU-c=GPU-d",
        )

    def test_everything_else_passes_through(self):
        self.assertEqual(self._run("-h"), "-h")
        self.assertEqual(self._run("--get-restore-tid", "--pid", "7"), "--get-restore-tid --pid 7")
        self.assertEqual(self._run("--get-state", "--pid", "7"), "--get-state --pid 7")
        self.assertEqual(
            self._run("--action", "lock", "--pid", "7", "--timeout", "5"),
            "--action lock --pid 7 --timeout 5",
        )
        self.assertEqual(self._run("--action", "unlock", "--pid", "7"), "--action unlock --pid 7")

    def test_an_explicit_map_is_not_doubled(self):
        self.assertEqual(
            self._run("--action", "restore", "--pid", "7", "--device-map", "x=y"),
            "--action restore --pid 7 --device-map x=y",
        )

    def test_without_the_variable_it_is_a_pass_through(self):
        env = dict(self.env)
        del env["MNCR_DEVICE_MAP"]
        self.assertEqual(
            self._run("--action", "restore", "--pid", "7", env=env),
            "--action restore --pid 7",
        )

    def test_rewritten_only_when_the_real_binary_moves(self):
        before = os.stat(self.shim).st_mtime_ns
        self.backend.shim_env("GPU-a=GPU-b")
        self.assertEqual(os.stat(self.shim).st_mtime_ns, before)

    def test_missing_real_binary_is_an_error(self):
        from mncr.errors import CriuError

        backend = CriuBackend(cuda_checkpoint=os.path.join(self.tmp, "nope"),
                              shim_dir=os.path.join(self.tmp, "shim2"))
        with self.assertRaises(CriuError):
            backend.shim_env("a=b")


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



class TestResumeUnknownState(unittest.TestCase):
    """resume() must not report success for a process it could not read."""

    class _Backend:
        def __init__(self, states, restore_error):
            self.states = list(states)
            self.restore_error = restore_error
            self.unlocked = False

        def get_state(self, pid):
            state = self.states.pop(0)
            if isinstance(state, Exception):
                raise state
            return state

        def restore(self, pid, device_map=None):
            raise DriverError(self.restore_error)

        def unlock(self, pid):
            self.unlocked = True

    def test_unknown_then_running_is_success(self):
        from agent.driver import ALREADY, resume

        backend = self._Backend([DriverError("no state"), STATE_RUNNING], ALREADY)
        self.assertEqual(resume(backend, 7), "already-running")

    def test_unknown_then_not_running_is_an_error(self):
        from agent.driver import ALREADY, resume

        backend = self._Backend([DriverError("no state"), "locked"], ALREADY)
        with self.assertRaises(DriverError):
            resume(backend, 7)

    def test_unknown_and_unreadable_is_an_error(self):
        from agent.driver import ALREADY, resume

        backend = self._Backend([DriverError("no state"), DriverError("still no state")], ALREADY)
        with self.assertRaises(DriverError):
            resume(backend, 7)


if __name__ == "__main__":
    unittest.main()
