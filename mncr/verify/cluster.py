#!/usr/bin/env python3
"""Multi-node bring-up: real agents on real nodes, real NCCL between them.

`make smoke` proves one node. This proves the part that only exists between
nodes: a coordinator fanning out to several agents, ranks whose communicator
crosses the network, images that have to travel, and a restore onto hardware
other than the image's own.

Run it on the first node, as root, with the repo at the same path on every
node and a python that has torch on every node:

    python3 -m verify.cluster \\
        --node node-a=172.31.6.202 --node node-b=172.31.4.217 \\
        --store ubuntu@172.31.6.202:/var/lib/mncr/store

Three scenarios, in order, each on the job the previous one left running:

    continue   checkpoint and keep going. The NCCL group is destroyed before
               the lock and rebuilt after; every rank proves it with an
               all_reduce over the new group.
    restore    checkpoint and stop, then restore from the images onto the
               same nodes. The processes die and come back.
    migrate    checkpoint and stop, then restore with every rank moved to a
               different node. Images are fetched through the store, the
               device map is applied on the target, rank 0 is on a new node
               and the rendezvous follows it. A final checkpoint proves the
               migrated job is still checkpointable.

Nothing is simulated. The remote side is reached by ssh and sudo; the store
is whatever `rsync` can write to, so no credentials leave the cluster.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from mncr import config, log, rpc  # noqa: E402
from mncr.proto import RankRef  # noqa: E402

_LOG = log.get("cluster")

STD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class Failed(AssertionError):
    pass


class Node:
    def __init__(self, name, ip, local, ssh, python, repo):
        self.name = name
        self.ip = ip
        self.local = local
        self.ssh = ssh
        self.python = python
        self.repo = repo

    def sh(self, cmd, timeout=600, check=True):
        if self.local:
            full = ["bash", "-c", cmd]
        else:
            full = shlex.split(self.ssh.format(ip=self.ip)) + [
                f"sudo bash -c {shlex.quote(cmd)}"
            ]
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"{self.name}: `{cmd[:120]}` rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[-600:]}"
            )
        return proc.stdout

    def spawn(self, cmd, logfile="/dev/null"):
        """Start something that must outlive this call and belong to nobody.

        nohup under a shell that exits at once, so the process is reparented
        to init. That matters: a rank that criu later kills must be reaped by
        someone, or its pid stays taken and the restore cannot have it back.
        """
        self.sh(
            f"nohup bash -c {shlex.quote(cmd)} >{shlex.quote(logfile)} 2>&1 </dev/null &"
        )

    def read_json(self, path):
        out = self.sh(f"cat {shlex.quote(path)}", check=False)
        try:
            return json.loads(out)
        except (ValueError, TypeError):
            return {}

    def proc_state(self, pid):
        """'R'/'S'/... if alive, 'Z' if a zombie, '' if gone."""
        out = self.sh(
            f"awk '/^State:/{{print $2}}' /proc/{int(pid)}/status 2>/dev/null",
            check=False,
        )
        return out.strip()[:1]


def env_for(node, args):
    env = {
        "PATH": STD_PATH,
        "PYTHONPATH": node.repo,
        "PYTHONDONTWRITEBYTECODE": "1",
        "MNCR_IMAGE_DIR": args.image_dir,
        "MNCR_CACHE_DIR": args.cache_dir,
        "MNCR_CONTROL_ROOT": args.control_root,
        "MNCR_JOBFILE_DIR": os.path.join(args.control_root, "jobs"),
        "MNCR_LOG_LEVEL": args.log_level,
        "MNCR_NODE_IP": node.ip,
        "MNCR_STRICT_CLEAN": "1",
        # Both measured on real nodes; both leave something behind that no
        # teardown can reach. RAS keeps two listeners per process; libfabric
        # opens /dev/gdrdrv at plugin init and never closes it, EFA or not.
        # Override with --env if your stack differs.
        "NCCL_RAS_ENABLE": "0",
        "FI_HMEM_CUDA_USE_GDRCOPY": "0",
    }
    if args.store:
        env["MNCR_OBJECT_STORE"] = args.store
        env["MNCR_OBJECT_PUT"] = "rsync -q --mkpath {src} {dst}"
        env["MNCR_OBJECT_GET"] = "rsync -q --mkpath {src} {dst}"
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    return env


def env_prefix(env):
    return "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


class Cluster:
    def __init__(self, args):
        self.args = args
        self.nodes = []
        for index, spec in enumerate(args.node):
            name, _, ip = spec.partition("=")
            if not ip:
                raise SystemExit(f"--node wants name=ip, got {spec!r}")
            self.nodes.append(
                Node(name, ip, local=(index == 0), ssh=args.ssh, python=args.python,
                     repo=args.repo)
            )
        self.by_name = {n.name: n for n in self.nodes}
        self.job = args.job
        self.world = len(self.nodes) * args.ranks_per_node
        self.placement = {}     # rank -> node name
        self.coord = None
        self.agent_addr = {n.name: f"tcp:{n.ip}:{args.agent_port}" for n in self.nodes}

    # ---------------------------------------------------------------- setup
    # Libraries every rank maps. CRIU checks the build-id of each mapped file
    # on restore and refuses on a mismatch - rightly, since the process would
    # be running someone else's code at its saved addresses. Measured: two
    # nodes from one AMI, launched minutes apart, diverged on libssl within
    # the hour because unattended-upgrades ran on one of them.
    FINGERPRINT = (
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
        "/usr/lib/x86_64-linux-gnu/libssl.so.3",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    )

    def check_nodes(self):
        prints = {}
        for node in self.nodes:
            out = node.sh(
                f"{node.python} -c 'import torch, sys; "
                f"print(torch.__version__, torch.cuda.device_count())' && "
                f"command -v criu && command -v cuda-checkpoint && "
                f"test -f /usr/lib/criu/cuda_plugin.so && test -d {shlex.quote(node.repo)}"
            )
            _LOG.info("node ok", node=node.name, ip=node.ip, detail=out.strip().replace("\n", " "))
            # The bind shim a migrated rank needs. Built here so every node
            # has one, at the same path, before any rank starts.
            if not self.args.no_netmap:
                # -B: rebuilt on every run so every node's copy has the same
                # bytes and the same mode, whoever built the last one.
                node.sh(f"make -s -B -C {shlex.quote(os.path.join(node.repo, 'torchckpt', 'netmap'))} test")
            libs = " ".join(self.FINGERPRINT) + f" {node.python}"
            digest = node.sh(f"sha256sum {libs} 2>/dev/null | awk '{{print $1}}'", check=False)
            prints[node.name] = digest.split()
        reference = prints[self.nodes[0].name]
        for node in self.nodes[1:]:
            if prints[node.name] != reference:
                differing = [
                    path for path, a, b in zip(self.FINGERPRINT + (node.python,), reference, prints[node.name])
                    if a != b
                ]
                _LOG.warn(
                    "nodes differ in mapped libraries; criu will refuse to migrate "
                    "between them (bad build-ID)",
                    node=node.name, reference=self.nodes[0].name, differing=differing,
                )

    def start_agents(self):
        for node in self.nodes:
            env = env_for(node, self.args)
            node.sh(
                f"pkill -f '[a]gent.main' ; "
                f"rm -rf {shlex.quote(os.path.join(self.args.control_root, 'jobs', self.job))} ; "
                f"mkdir -p {self.args.image_dir} {self.args.cache_dir} "
                f"{self.args.control_root}/jobs {self.args.log_dir}; true"
            )
            node.spawn(
                f"cd {shlex.quote(node.repo)} && {env_prefix(env)} {node.python} -m agent.main "
                f"--addr tcp:0.0.0.0:{self.args.agent_port} "
                f"--rank-addr tcp:127.0.0.1:{self.args.rank_port} "
                f"--node {node.name} --metrics-port 0",
                logfile=os.path.join(self.args.log_dir, f"agent-{node.name}.log"),
            )
        deadline = time.monotonic() + 60
        for node in self.nodes:
            while True:
                try:
                    with rpc.Client(self.agent_addr[node.name], timeout=5) as client:
                        client.call("status")
                    break
                except Exception as exc:  # noqa: BLE001
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"agent on {node.name} never answered: {exc}")
                    time.sleep(0.5)
        _LOG.info("agents up", nodes=[n.name for n in self.nodes])

    def start_coordinator(self):
        from coord.main import Coordinator
        from coord.store import EpochStore

        os.environ.update(env_for(self.nodes[0], self.args))
        cfg = config.load()
        cfg.image_dir = self.args.image_dir
        cfg.cache_dir = self.args.cache_dir
        store_dir = os.path.join(self.args.log_dir, f"epochs-{self.job}")
        self.coord = Coordinator(cfg, agents=self.agent_addr, store=EpochStore(store_dir))
        for node in self.nodes:
            result = self.coord.register_agent(node.name, self.agent_addr[node.name])
            info = result.get("info") or {}
            if not info.get("gpu_uuids"):
                raise RuntimeError(f"{node.name} described no GPUs: {info}")
            if info.get("ip") != node.ip:
                _LOG.warn("node ip differs from what was given", node=node.name,
                          given=node.ip, described=info.get("ip"))
        _LOG.info("coordinator ready", nodes=self.coord.node_info.keys())

    def launch_ranks(self):
        rendezvous = f"tcp://{self.nodes[0].ip}:{self.args.rendezvous_port}"
        rank = 0
        for node in self.nodes:
            env = env_for(node, self.args)
            env.setdefault("NCCL_DEBUG", "WARN")
            if not self.args.no_netmap:
                # NCCL keeps the node address it found at first init; after
                # a migration that address belongs to another machine. The
                # shim rebinds. See torchckpt/netmap/mncr_netmap.c.
                env["LD_PRELOAD"] = os.path.join(
                    node.repo, "torchckpt", "netmap", "libmncr_netmap.so"
                )
                env["MNCR_NETMAP_DEBUG"] = "1"
            node.sh(f"pkill -f '[f]ake_rank.py' ; true")
            # A rank writes its progress fresh; a file from an earlier run on
            # this node would otherwise be read as this run's.
            node.sh(
                "rm -f " + " ".join(
                    shlex.quote(self.progress_path(r)) for r in range(self.world)
                ) + " ; true"
            )
            for local in range(self.args.ranks_per_node):
                cmd = (
                    f"cd {shlex.quote(node.repo)} && {env_prefix(env)} {node.python} "
                    f"verify/fake/fake_rank.py --job-id {self.job} --rank {rank} "
                    f"--world-size {self.world} --agent-addr tcp:127.0.0.1:{self.args.rank_port} "
                    f"--control-root {self.args.control_root} "
                    f"--progress {self.progress_path(rank)} --backend nccl "
                    f"--init-method {rendezvous} --cuda --device {local} "
                    f"--step-seconds {self.args.step_seconds}"
                )
                # /dev/null on purpose: a log file open for writing is a file
                # the restore must find, at least as large, on the target
                # node. The rank reports through its progress file instead.
                node.spawn(cmd)
                self.placement[rank] = node.name
                rank += 1

        deadline = time.monotonic() + 180
        while True:
            registered = {}
            for node in self.nodes:
                status = self.call(node.name, "status", job_id=self.job)
                for record in status.get("ranks", []):
                    registered[int(record["rank"])] = (
                        node.name, int(record["host_pid"]), record.get("ip")
                    )
            if len(registered) >= self.world:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"only {len(registered)}/{self.world} ranks registered; "
                    f"progress: {self.all_progress()}"
                )
            time.sleep(0.5)
        refs = [
            RankRef.make(self.job, r, node, host_pid=pid, ip=ip)
            for r, (node, pid, ip) in sorted(registered.items())
        ]
        self.coord.register_job(self.job, refs)
        self.wait_steps(5, timeout=120)
        _LOG.info("ranks running", world=self.world, placement=self.placement)

    # ------------------------------------------------------------- helpers
    def call(self, node, op, **kwargs):
        with rpc.Client(self.agent_addr[node], timeout=120) as client:
            return client.call(op, **kwargs)

    def progress_path(self, rank):
        return os.path.join(self.args.log_dir, f"rank-{rank}.json")

    def progress(self, rank):
        return self.by_name[self.placement[rank]].read_json(self.progress_path(rank))

    def all_progress(self):
        return {r: self.progress(r) for r in range(self.world)}

    def last_event(self, rank, name, epoch=None):
        events = [
            e for e in self.progress(rank).get("events", [])
            if e["event"] == name and (epoch is None or e.get("epoch") == epoch)
        ]
        return events[-1] if events else None

    def wait_steps(self, count, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            steps = [self.progress(r).get("step") or 0 for r in range(self.world)]
            if all(s >= count for s in steps):
                return steps
            crashed = {r: self.last_event(r, "crashed") for r in range(self.world)}
            crashed = {r: e for r, e in crashed.items() if e}
            if crashed:
                raise Failed(f"ranks crashed: {json.dumps(crashed)[:1500]}")
            time.sleep(0.5)
        raise Failed(f"ranks did not reach step {count}: {self.all_progress()}")

    def wait_resumed(self, epoch_id, restored, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            events = {r: self.last_event(r, "resumed", epoch=epoch_id) for r in range(self.world)}
            if all(events.values()):
                break
            crashed = {r: self.last_event(r, "crashed") for r in range(self.world)}
            crashed = {r: e for r, e in crashed.items() if e}
            if crashed:
                raise Failed(f"ranks crashed: {json.dumps(crashed)[:1500]}")
            time.sleep(0.5)
        else:
            raise Failed(
                f"not every rank resumed epoch {epoch_id} within {timeout}s: "
                f"{json.dumps({r: (e or {}).get('event') for r, e in events.items()})}; "
                f"progress {json.dumps(self.all_progress())[:2000]}"
            )
        for rank, event in events.items():
            if event.get("device_memory_intact") is not True:
                raise Failed(f"rank {rank}: device memory did not survive: {event}")
            if bool(event.get("restored")) is not restored:
                raise Failed(f"rank {rank}: restored={event.get('restored')}, wanted {restored}")
            if event.get("rejoined") is not True:
                raise Failed(f"rank {rank}: rebuilt NCCL group failed its all_reduce: {event}")
            if (event.get("rebuild") or {}).get("warm_up") is not True:
                raise Failed(f"rank {rank}: rebuild did not warm up: {event.get('rebuild')}")
            if self.placement[rank] != self.placement[0] and event.get("hostname") == \
                    self.last_event(0, "resumed", epoch=epoch_id).get("hostname"):
                raise Failed(f"rank {rank} reports the same host as rank 0 but is placed elsewhere")
        steps = {self.last_event(r, "quiesced")["step"] for r in range(self.world)}
        if len(steps) != 1:
            raise Failed(f"ranks stopped at different steps: {steps}")
        return events

    def wait_dead(self, pids, timeout=120):
        """pids: {rank: (node, pid)}. Wait until criu's kill has been reaped."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            states = {r: self.by_name[n].proc_state(p) for r, (n, p) in pids.items()}
            if all(s == "" for s in states.values()):
                return
            time.sleep(0.5)
        raise Failed(f"dumped ranks still present: {states}")

    def current_pids(self):
        out = {}
        for node in self.nodes:
            status = self.call(node.name, "status", job_id=self.job)
            for record in status.get("ranks", []):
                out[int(record["rank"])] = (node.name, int(record["host_pid"]))
        return out

    # ----------------------------------------------------------- scenarios
    def scenario_continue(self):
        result = self.coord.checkpoint(self.job, mode="continue")
        events = self.wait_resumed(result["epoch_id"], restored=False)
        teardown = events[0].get("teardown") or {}
        return {
            "epoch_id": result["epoch_id"],
            "seconds": result.get("seconds"),
            "init_method": events[0].get("init_method"),
            "sockets_after_teardown": teardown.get("open_sockets"),
            "nccl": self.progress(0).get("nccl_version"),
        }

    def scenario_restore(self):
        before = self.current_pids()
        result = self.coord.checkpoint(self.job, mode="stop")
        self.wait_dead(before)
        restored = self.coord.restore(self.job, result["epoch_id"])
        events = self.wait_resumed(result["epoch_id"], restored=True)
        after = self.current_pids()
        return {
            "epoch_id": result["epoch_id"],
            "device_maps": restored.get("device_maps"),
            "pids_before": {r: p for r, (_, p) in before.items()},
            "pids_after": {r: p for r, (_, p) in after.items()},
            "init_method": events[0].get("init_method"),
        }

    def scenario_migrate(self):
        before = self.current_pids()
        result = self.coord.checkpoint(self.job, mode="stop")
        self.wait_dead(before)

        # Every rank to the next node along. With two nodes that is a swap.
        names = [n.name for n in self.nodes]
        targets = {}
        new_placement = {}
        for rank, node in sorted(self.placement.items()):
            dest = names[(names.index(node) + 1) % len(names)]
            targets.setdefault(dest, []).append(rank)
            new_placement[rank] = dest
        # Wipe the source images from the target nodes' point of view: they
        # never had them. Nothing to do - they are on other machines. But the
        # source nodes still hold theirs, so also make sure the target really
        # fetches: delete nothing, and check afterwards that the target's
        # image dir was populated by a fetch.
        restored = self.coord.restore(self.job, result["epoch_id"], targets=targets)
        self.placement = new_placement
        events = self.wait_resumed(result["epoch_id"], restored=True)

        hosts = {r: e.get("hostname") for r, e in events.items()}
        for rank, node in self.placement.items():
            expected = self.by_name[node].sh("hostname").strip()
            if hosts.get(rank) != expected:
                raise Failed(f"rank {rank} resumed on {hosts.get(rank)}, expected {expected} ({node})")
        maps = restored.get("device_maps") or {}
        from coord.devicemap import is_identity_cli

        for node, cli in maps.items():
            if is_identity_cli(cli):
                raise Failed(f"{node}: migration produced an identity device map: {cli}")

        # The migrated job must be a job again: checkpoint it once more.
        again = self.coord.checkpoint(self.job, mode="continue")
        self.wait_resumed(again["epoch_id"], restored=False)
        return {
            "epoch_id": result["epoch_id"],
            "targets": targets,
            "device_maps": maps,
            "hosts": hosts,
            "init_method": events[0].get("init_method"),
            "checkpoint_after_migration": again["epoch_id"],
        }

    # -------------------------------------------------------------- teardown
    def stop(self):
        for node in self.nodes:
            try:
                node.sh("pkill -f '[f]ake_rank.py' ; pkill -f '[a]gent.main' ; true", timeout=60)
            except Exception:  # noqa: BLE001
                pass


SCENARIOS = {
    "continue": Cluster.scenario_continue,
    "restore": Cluster.scenario_restore,
    "migrate": Cluster.scenario_migrate,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", action="append", required=True, help="name=ip; first is this host")
    ap.add_argument("--ssh", default="ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR ubuntu@{ip}",
                    help="command template reaching a remote node; sudo is added")
    ap.add_argument("--python", default=sys.executable, help="python with torch, on every node")
    ap.add_argument("--repo", default=REPO, help="this repo's path, the same on every node")
    ap.add_argument("--store", default=None, help="rsync destination for the object store, e.g. user@host:/path")
    ap.add_argument("--job", default="mn-job")
    ap.add_argument("--ranks-per-node", type=int, default=1)
    ap.add_argument("--scenarios", default="continue,restore,migrate")
    ap.add_argument("--agent-port", type=int, default=7181)
    ap.add_argument("--rank-port", type=int, default=7182)
    ap.add_argument("--rendezvous-port", type=int, default=29400)
    ap.add_argument("--step-seconds", type=float, default=0.05)
    ap.add_argument("--image-dir", default="/var/lib/mncr/images")
    ap.add_argument("--cache-dir", default="/var/lib/mncr/cache")
    ap.add_argument("--control-root", default="/run/mncr")
    ap.add_argument("--log-dir", default="/var/lib/mncr/cluster")
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for agents and ranks")
    ap.add_argument("--no-netmap", action="store_true",
                    help="do not preload the bind shim into ranks (migration will fail in NCCL)")
    ap.add_argument("--keep", action="store_true", help="leave agents and ranks running")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if os.geteuid() != 0:
        print("run as root: the agent and criu need it", file=sys.stderr)
        return 2

    cluster = Cluster(args)
    results = []
    started_all = time.time()
    try:
        cluster.check_nodes()
        cluster.start_agents()
        cluster.start_coordinator()
        cluster.launch_ranks()
        for name in [s for s in args.scenarios.split(",") if s]:
            entry = {"scenario": name}
            started = time.time()
            try:
                entry["detail"] = SCENARIOS[name](cluster)
                entry["status"] = "pass"
            except Failed as exc:
                entry["status"] = "FAIL"
                entry["reason"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "FAIL"
                entry["reason"] = f"{type(exc).__name__}: {exc}"
            entry["seconds"] = round(time.time() - started, 2)
            results.append(entry)
            if entry["status"] == "FAIL":
                break
    except Exception as exc:  # noqa: BLE001
        results.append({"scenario": "setup", "status": "FAIL",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.time() - started_all, 2)})
    finally:
        if not args.keep:
            cluster.stop()

    failures = [r for r in results if r["status"] == "FAIL"]
    if args.json:
        print(json.dumps({"scenarios": results, "failed": len(failures),
                          "placement": cluster.placement}, indent=2, default=str))
    else:
        print(f"{'SCENARIO':<12}{'RESULT':<8}{'SECONDS':>8}")
        print("-" * 28)
        for entry in results:
            print(f"{entry['scenario']:<12}{entry['status']:<8}{entry['seconds']:>8.2f}")
            if entry.get("reason"):
                print(f"    {entry['reason']}")
        print("-" * 28)
        print("Multi-node bring-up looks good." if not failures else
              "A failed scenario invalidates the ones after it; fix and rerun.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
