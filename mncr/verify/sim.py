"""In-process cluster simulator.

Runs real agents, a real coordinator and real rank processes with the fake
driver and fake CRIU behind them. Every protocol test and the chaos matrix build
on this: the parts under test are the ones that would otherwise only be
exercisable on a GPU cluster.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from mncr import config
from mncr.proto import RankRef

from agent.main import Agent, build_server as build_agent_server
from coord.main import Coordinator
from coord.store import EpochStore

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_RANK = os.path.join(HERE, "fake", "fake_rank.py")


class SimCluster:
    def __init__(self, nodes=2, ranks_per_node=2, job_id="sim-job", step_seconds=0.005,
                 call_latency=0.0, real=False):
        self.nodes = nodes
        self.ranks_per_node = ranks_per_node
        self.job_id = job_id
        self.step_seconds = step_seconds
        self.call_latency = call_latency
        # real=True swaps in the CLI driver and CRIU backends. Same protocol,
        # same code path, actual hardware underneath - this is what turns the
        # simulator into the on-node integration test.
        self.real = real
        self.gloo = False
        self.root = tempfile.mkdtemp(prefix="mncr-sim-")
        self.control_root = os.path.join(self.root, "control")
        self.image_root = os.path.join(self.root, "images")
        self.agents = {}
        self.servers = []
        self.procs = []
        self.coord = None
        os.makedirs(self.control_root, exist_ok=True)
        os.makedirs(self.image_root, exist_ok=True)

    # ------------------------------------------------------------------ setup
    def start(self):
        cfg = config.Config()
        cfg.fake = not self.real
        cfg.image_dir = self.image_root
        cfg.cache_dir = os.path.join(self.root, 'cache')
        cfg.quiesce_timeout = 30.0
        cfg.checkpoint_timeout = 60.0
        cfg.strict_clean = True
        cfg.fake_call_latency = self.call_latency
        if not self.real:
            # The fake driver releases nothing, so on a GPU node the agent's
            # dump gate would find every fd a real rank holds. Ranks already
            # scan a non-existent root; the agent does the same.
            cfg.proc_root = os.path.join(self.root, "no-proc")
        self.cfg = cfg

        node_info = {}
        for index in range(self.nodes):
            node = f"node-{index}"
            node_control = os.path.join(self.control_root, node)
            os.makedirs(node_control, exist_ok=True)
            agent = Agent(cfg, node_name=node, control_root=node_control)
            server = build_agent_server(agent, "tcp:127.0.0.1:0").start()
            self.servers.append(server)
            self.agents[node] = agent
            addr = f"tcp:127.0.0.1:{server.port}"
            agent.rpc_addr = addr
            if self.real:
                # The real driver will be asked to restore with whatever map
                # this produces, so the UUIDs have to be the node's own.
                node_info[node] = {**agent.describe(), "host": node, "ip": "127.0.0.1"}
            else:
                node_info[node] = {
                    "host": node,
                    "ip": "127.0.0.1",
                    "gpu_count": self.ranks_per_node,
                    "gpu_names": "|".join(["H100"] * self.ranks_per_node),
                    "gpu_uuids": "|".join(
                        f"GPU-{index:04d}{g:04d}-0000-0000-0000-000000000000"
                        for g in range(self.ranks_per_node)
                    ),
                    "driver_version": "610.57.04",
                    "gpu_mem_total_mib": 81920 * self.ranks_per_node,
                    "ram_headroom_mib": 2000000,
                    "criu_cuda_plugin": True,
                    "mnnvl": False,
                }

        self.coord = Coordinator(
            cfg,
            agents={n: a.rpc_addr for n, a in self.agents.items()},
            store=EpochStore(os.path.join(self.root, "epochs")),
            node_info=node_info,
        )
        return self

    def launch_ranks(self, dirty_ranks=(), cuda=None, gloo=False, backend=None):
        """backend: None (no process group), "gloo" (CPU), or "nccl" (real GPUs,
        one per rank on this host)."""
        dirty = set(dirty_ranks)
        cuda = self.real if cuda is None else cuda
        backend = backend or ("gloo" if gloo else None)
        self.gloo = backend is not None
        self.backend = backend
        init_method = self.new_rendezvous() if backend else None
        world = self.nodes * self.ranks_per_node
        rank_id = 0
        for index in range(self.nodes):
            node = f"node-{index}"
            addr = self.agents[node].rpc_addr
            node_control = self.agents[node].control_root
            for local in range(self.ranks_per_node):
                progress = os.path.join(self.root, f"rank-{rank_id}.json")
                cmd = [
                    sys.executable, FAKE_RANK,
                    "--job-id", self.job_id,
                    "--rank", str(rank_id),
                    "--world-size", str(world),
                    "--agent-addr", addr,
                    "--control-root", node_control,
                    "--progress", progress,
                    "--step-seconds", str(self.step_seconds),
                    "--proc-root", "/proc" if self.real else os.path.join(self.root, "no-proc"),
                ]
                if rank_id in dirty:
                    cmd.append("--dirty")
                if cuda:
                    cmd.append("--cuda")
                if backend:
                    cmd += ["--backend", backend, "--init-method", init_method]
                if backend == "nccl":
                    cmd += ["--device", str(local)]
                env = dict(os.environ, MNCR_LOG_LEVEL="warn", PYTHONPATH=_repo_root())
                self.procs.append(
                    (rank_id, node, subprocess.Popen(cmd, env=env,
                                                     stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.PIPE))
                )
                rank_id += 1
        return self

    def wait_registered(self, timeout=30.0):
        want = self.nodes * self.ranks_per_node
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            have = sum(len(a.local_ranks(self.job_id)) for a in self.agents.values())
            if have >= want:
                self.coord.register_job(self.job_id, self.rank_refs())
                return True
            time.sleep(0.05)
        raise TimeoutError(
            f"only {have}/{want} ranks registered after {timeout}s; "
            f"rank stderr: {self.rank_stderr()}"
        )

    def rank_refs(self):
        refs = []
        for node, agent in self.agents.items():
            for record in agent.local_ranks(self.job_id):
                refs.append(
                    RankRef.make(
                        self.job_id, record["rank"], node, host_pid=record["host_pid"]
                    )
                )
        return sorted(refs, key=lambda r: r["rank"])

    # ------------------------------------------------------------ inspection
    def new_rendezvous(self):
        """A fresh rendezvous address.

        Fresh per epoch on purpose: after a restore the peers are at different
        addresses, so reusing the old one would test nothing and would collide
        with the store the destroyed group left behind.

        File-based rather than TCP. Picking a free port by binding to zero and
        closing leaves a window in which something else can take it, and under
        the load of a full test run that window gets hit - the symptom is an
        occasional rendezvous timeout with nothing wrong in the code under test.
        A path cannot be stolen.
        """
        self._rendezvous_seq = getattr(self, "_rendezvous_seq", 0) + 1
        path = os.path.join(self.root, f"rendezvous-{self._rendezvous_seq}")
        return f"file://{path}"

    def progress(self, rank):
        path = os.path.join(self.root, f"rank-{rank}.json")
        try:
            with open(path) as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def all_progress(self):
        return [self.progress(r) for r, _, _ in self.procs]

    def wait_for_event(self, event, count=None, timeout=30.0):
        count = count if count is not None else len(self.procs)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            seen = sum(
                1
                for p in self.all_progress()
                if any(e["event"] == event for e in p.get("events", []))
            )
            if seen >= count:
                return True
            time.sleep(0.05)
        return False

    def last_event(self, rank, event):
        events = [e for e in self.progress(rank).get("events", []) if e["event"] == event]
        return events[-1] if events else None

    def driver_calls(self):
        return {n: list(a.driver.calls) for n, a in self.agents.items()}

    def alive(self):
        return sum(1 for _, _, p in self.procs if p.poll() is None)

    def wait_for_exit(self, timeout=60.0):
        """Reap every launched rank.

        After a checkpoint-and-stop the ranks are dead - criu killed them -
        but as our children they sit as zombies until waited on, and a zombie
        still owns its pid. criu restore needs that pid back.
        """
        for _, _, proc in self.procs:
            proc.wait(timeout=timeout)
        return True

    def registered_pids(self):
        pids = set()
        for agent in self.agents.values():
            for record in agent.local_ranks(self.job_id):
                pids.add(int(record["host_pid"]))
        return pids

    def rank_stderr(self, rank=None):
        """Stderr of exited ranks, so a dead rank explains itself."""
        out = {}
        for rank_id, _node, proc in self.procs:
            if rank is not None and rank_id != rank:
                continue
            if proc.poll() is not None and proc.stderr is not None:
                try:
                    out[rank_id] = proc.stderr.read().decode(errors="replace")[-2000:]
                except (ValueError, OSError):
                    out[rank_id] = "(stderr already consumed)"
        return out

    # -------------------------------------------------------------- teardown
    def stop(self):
        import signal

        own = {p.pid for _, _, p in self.procs}
        # Ranks that came back through criu restore are nobody's children;
        # the agents' registries are the only record of them.
        for pid in self.registered_pids() - own:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for _, _, proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for _, _, proc in self.procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
        for server in self.servers:
            server.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _repo_root():
    return os.path.dirname(HERE)
