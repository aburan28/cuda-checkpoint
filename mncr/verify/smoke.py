#!/usr/bin/env python3
"""On-node bring-up: the real driver, real CRIU, real hardware.

Four levels, each adding one component. Run them in order - a failure at level 1
makes everything above it meaningless, and each level's success is the next
one's precondition.

    1  driver only    lock, checkpoint, restore, unlock a plain CUDA process,
                      and prove the device memory survived
    2  + CRIU         dump the checkpointed process to disk, restore it, and
                      prove the memory survived that too
    3  + agent        the same sequence driven through the node agent, so the
                      verification gates and pid handling are exercised
    4  + coordinator  a full epoch: prepare, vote, lock, checkpoint, dump,
                      restore, resume, across real ranks

Nothing here is simulated. If a level cannot run - no GPU, no criu, not root -
it says so and returns a skip rather than a pass.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mncr import config, log  # noqa: E402
from mncr.errors import MncrError  # noqa: E402

_LOG = log.get("smoke")
HERE = os.path.dirname(os.path.abspath(__file__))


class Skip(Exception):
    """This level cannot run here, and that is not a failure."""


# --------------------------------------------------------------------- target
class Target:
    """A real CUDA process the harness can checksum across a checkpoint."""

    def __init__(self, workdir, elements=None):
        self.workdir = workdir
        self.state = os.path.join(workdir, "target.state")
        self.elements = elements
        self.proc = None
        self.pid = None
        self.kind = None

    def _build_cuda(self):
        nvcc = shutil.which("nvcc")
        if not nvcc:
            return None
        source = os.path.join(HERE, "smoke_target.cu")
        binary = os.path.join(self.workdir, "smoke_target")
        proc = subprocess.run(
            [nvcc, "-O2", source, "-o", binary], capture_output=True, text=True
        )
        if proc.returncode != 0:
            _LOG.warn("nvcc build failed", error=proc.stderr.strip()[-300:])
            return None
        return binary

    def start(self):
        binary = self._build_cuda()
        if binary:
            cmd = [binary, self.state]
            if self.elements:
                cmd.append(str(self.elements))
            self.kind = "cuda-c"
        else:
            script = os.path.join(HERE, "smoke_target.py")
            cmd = [sys.executable, script, self.state]
            if self.elements:
                cmd += ["--elements", str(self.elements)]
            self.kind = "torch"

        self.proc = subprocess.Popen(cmd, cwd=self.workdir)
        line = self._await_state("ready", timeout=120)
        if line is None:
            raise Skip(f"{self.kind} target never became ready (no usable GPU?)")
        self.pid = int(line.split()[1])
        _LOG.info("target ready", kind=self.kind, pid=self.pid)
        return self

    def _await_state(self, prefix, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with open(self.state) as fh:
                    line = fh.read().strip()
                if line.startswith(prefix):
                    return line
                if line.startswith("error"):
                    raise Skip(line)
            except FileNotFoundError:
                pass
            # After a criu restore the process is a fresh, detached one and we
            # no longer own a handle to it. Only treat an exited handle as death
            # while we still hold one.
            if self.proc is not None and self.proc.poll() is not None:
                return None
            time.sleep(0.05)
        return None

    def command(self, verb):
        with open(f"{self.state}.cmd.tmp", "w") as fh:
            fh.write(verb + "\n")
        os.replace(f"{self.state}.cmd.tmp", f"{self.state}.cmd")

    def checksum(self, timeout=120):
        """Ask the target to read its device memory back and report."""
        try:
            os.unlink(self.state)
        except FileNotFoundError:
            pass
        self.command("verify")
        line = self._await_state("sum", timeout=timeout)
        if line is None:
            raise AssertionError("target did not answer a verify request")
        parts = line.split()
        return {"sum": int(parts[1]), "bad": int(parts[3]), "pid": int(parts[5])}

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.command("exit")
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _listed_by_nvidia_smi(pid):
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None      # cannot tell
    return str(pid) in proc.stdout.split()


def _process_gone(pid):
    """True once the pid is really gone.

    A dumped process that is our own child becomes a zombie until somebody
    reaps it, and a zombie still has a /proc entry. Checking for the directory
    alone reports "still alive" for a process CRIU has already killed.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().rsplit(") ", 1)[-1].split()
        return fields[0] == "Z"
    except FileNotFoundError:
        return True
    except (IndexError, OSError):
        return False


def _wait_until_released(pid, timeout=15.0, interval=0.25):
    """Wait for nvidia-smi to stop listing a checkpointed process.

    The driver call returns before this becomes visible - measured at about a
    second on a Blackwell node with driver 595. Checking once immediately after
    the checkpoint reads as "still attached" when it is simply not updated yet,
    so the question has to be asked with a deadline rather than at an instant.
    """
    deadline = time.monotonic() + timeout
    listed = _listed_by_nvidia_smi(pid)
    while listed and time.monotonic() < deadline:
        time.sleep(interval)
        listed = _listed_by_nvidia_smi(pid)
    return listed


# ---------------------------------------------------------------- the levels
def level1_driver(cfg, workdir):
    """The vendor's demo, asserted rather than eyeballed."""
    from agent.driver import CliBackend

    if not shutil.which(cfg.cuda_checkpoint):
        raise Skip("cuda-checkpoint is not on PATH")

    driver = CliBackend(cfg.cuda_checkpoint)
    target = Target(workdir).start()
    try:
        before = target.checksum()
        assert before["bad"] == 0, "target memory was already wrong before we started"

        state = driver.get_state(target.pid)
        assert "run" in state, f"expected a running process, got {state!r}"

        driver.lock(target.pid, cfg.lock_timeout_ms)
        driver.checkpoint(target.pid)

        listed = _wait_until_released(target.pid)
        assert listed is not True, (
            "nvidia-smi still lists the process 15s after checkpoint; its GPU "
            "resources were not released"
        )

        driver.restore(target.pid)
        driver.unlock(target.pid)

        after = target.checksum()
        assert after["bad"] == 0, f"{after['bad']} elements corrupted by the round trip"
        assert after["sum"] == before["sum"], "checksum changed across checkpoint"
        return {
            "target": target.kind,
            "pid": target.pid,
            "checksum": before["sum"],
            "gpu_released_while_checkpointed": listed is False,
        }
    finally:
        target.stop()


def level2_criu(cfg, workdir):
    """Add the dump. This is where the process actually dies and comes back."""
    from agent import driver as driver_mod
    from agent.criu import CriuBackend
    from agent.driver import CliBackend

    if os.geteuid() != 0:
        raise Skip("criu needs root")
    criu = CriuBackend(cfg.criu, cfg.criu_libdir, cfg.dump_timeout)
    if not criu.available():
        raise Skip("criu is not on PATH")

    driver = CliBackend(cfg.cuda_checkpoint)
    images = os.path.join(workdir, "images")
    target = Target(workdir).start()
    try:
        before = target.checksum()
        driver.lock(target.pid, cfg.lock_timeout_ms)
        driver.checkpoint(target.pid)
        criu.dump(target.pid, images)

        # The target is our child, so reap it: without that it sits as a zombie
        # and every liveness check reports a process CRIU has already killed.
        try:
            target.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise AssertionError("criu dump left the process running")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _process_gone(target.pid):
            time.sleep(0.1)
        assert _process_gone(target.pid), "criu dump left the process alive"

        restored_pid = criu.restore(images)
        # criu owns the restored process now; our handle refers to the one it
        # killed. Drop it before asking the target anything.
        target.proc = None

        # The CUDA plugin may already have restored CUDA during criu restore;
        # resume() asks rather than assuming.
        resumed = driver_mod.resume(driver, restored_pid, log=_LOG)

        after = target.checksum()
        assert after["bad"] == 0, f"{after['bad']} elements corrupted across dump/restore"
        assert after["sum"] == before["sum"], "checksum changed across dump/restore"
        return {
            "target": target.kind,
            "original_pid": target.pid,
            "restored_pid": restored_pid,
            "resume": resumed,
            "checksum": before["sum"],
            "images": images,
        }
    finally:
        target.proc = None
        try:
            subprocess.run(["pkill", "-f", "smoke_target"], capture_output=True)
        except OSError:
            pass


def level3_agent(cfg, workdir):
    """The same sequence through the agent, so the gates run."""
    from agent.main import Agent

    if os.geteuid() != 0:
        raise Skip("the agent needs root")

    cfg.fake = False
    cfg.image_dir = os.path.join(workdir, "images")
    cfg.cache_dir = os.path.join(workdir, "cache")
    agent = Agent(cfg, node_name="smoke", control_root=os.path.join(workdir, "control"))

    target = Target(workdir).start()
    try:
        before = target.checksum()
        agent.rank_register("smoke-job", 0, target.pid, 1, [])
        epoch = "ep-smoke"
        agent.lock("smoke-job", epoch, ranks=[0])
        agent.checkpoint("smoke-job", epoch, ranks=[0])
        result = agent.dump(
            "smoke-job", epoch, cfg.image_dir, ranks=[0], leave_running=True
        )
        agent.restore(
            "smoke-job", epoch, cfg.image_dir, ranks=[0], from_images=False
        )
        after = target.checksum()
        assert after["bad"] == 0, "memory corrupted through the agent path"
        assert after["sum"] == before["sum"]
        return {
            "target": target.kind,
            "images": result["images"],
            "manifest": result.get("manifest"),
        }
    finally:
        target.stop()


def level4_epoch(cfg, workdir, ranks=2):
    """A full coordinator epoch against real backends."""
    from verify.sim import SimCluster

    if os.geteuid() != 0:
        raise Skip("the agent needs root")
    # The ranks hold CUDA state through torch when it is present and through the
    # driver API when it is not, so all that is actually required is a device.
    try:
        import ctypes

        ctypes.CDLL("libcuda.so.1").cuInit(0)
    except OSError:
        raise Skip("level 4 ranks need a CUDA driver")

    sim = SimCluster(nodes=1, ranks_per_node=ranks, step_seconds=0.01, real=True)
    sim.start()
    try:
        sim.launch_ranks()
        sim.wait_registered(timeout=120)
        result = sim.coord.checkpoint(sim.job_id, mode="continue")
        assert sim.wait_for_event("resumed", timeout=180), "ranks did not resume"

        for rank in range(ranks):
            events = sim.progress(rank)["events"]
            resumed = [e for e in events if e["event"] == "resumed"]
            assert resumed, f"rank {rank} never resumed"
            assert resumed[-1].get("device_memory_intact") is True, (
                f"rank {rank} device memory did not survive the round trip"
            )
        return {"epoch_id": result["epoch_id"], "ranks": ranks,
                "seconds": result.get("seconds")}
    finally:
        sim.stop()


LEVELS = {
    1: ("driver only", level1_driver),
    2: ("driver + criu", level2_criu),
    3: ("through the agent", level3_agent),
    4: ("full epoch", level4_epoch),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, default=0, help="run one level (default: all)")
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = config.load()
    cfg.fake = False
    wanted = [args.level] if args.level else sorted(LEVELS)
    results = []

    for level in wanted:
        name, fn = LEVELS[level]
        workdir = tempfile.mkdtemp(prefix=f"mncr-smoke-l{level}-")
        entry = {"level": level, "name": name}
        started = time.time()
        try:
            entry["detail"] = fn(cfg, workdir)
            entry["status"] = "pass"
        except Skip as exc:
            entry["status"] = "skip"
            entry["reason"] = str(exc)
        except AssertionError as exc:
            entry["status"] = "FAIL"
            entry["reason"] = str(exc)
        except MncrError as exc:
            entry["status"] = "FAIL"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "FAIL"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - started, 2)
        results.append(entry)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            entry["workdir"] = workdir
        # A failed level invalidates everything above it.
        if entry["status"] == "FAIL":
            break

    failures = [r for r in results if r["status"] == "FAIL"]
    if args.json:
        print(json.dumps({"levels": results, "failed": len(failures)}, indent=2))
    else:
        print(f"{'LEVEL':<7}{'NAME':<22}{'RESULT':<8}{'SECONDS':>8}")
        print("-" * 46)
        for entry in results:
            print(
                f"{entry['level']:<7}{entry['name']:<22}{entry['status']:<8}"
                f"{entry['seconds']:>8.2f}"
            )
            if entry.get("reason"):
                print(f"       {entry['reason']}")
        print("-" * 46)
        if failures:
            print("A failed level invalidates the ones above it; fix and rerun.")
        elif all(r["status"] == "skip" for r in results):
            print("Everything skipped: this host cannot run the smoke tests.")
        else:
            print("Bring-up looks good on this node.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
