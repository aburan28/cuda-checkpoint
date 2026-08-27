"""Agent pool: calls every node in parallel and reports per-node outcomes.

Node-level parallelism is the whole reason checkpoint wall clock does not grow
with job size. The driver serializes within a node; nothing serializes across
them, so the fan-out here is what keeps a 512-rank job costing the same as a
64-rank one.
"""

import concurrent.futures
import threading

from mncr import log, rpc

_LOG = log.get("coord.client")


class AgentPool:
    def __init__(self, agents, timeout=900.0, max_workers=64):
        """agents: {node_name: address}"""
        self.agents = dict(agents)
        self.timeout = timeout
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def nodes(self):
        return sorted(self.agents)

    def add(self, node, addr):
        with self._lock:
            self.agents[node] = addr

    def call(self, node, op, _timeout=None, **args):
        addr = self.agents[node]
        with rpc.Client(addr, timeout=_timeout or self.timeout) as client:
            return client.call(op, **args)

    def fanout(self, nodes, op, per_node_args=None, _timeout=None, **common):
        """Run `op` on every node concurrently.

        Returns (results, errors) keyed by node. Never raises for a node
        failure - the caller decides what a partial failure means, and that
        decision depends on which side of the commit point it happened.
        """
        nodes = list(nodes)
        per_node_args = per_node_args or {}
        results, errors = {}, {}

        if not nodes:
            return results, errors

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(nodes)), thread_name_prefix="fanout"
        ) as pool:
            futures = {
                pool.submit(
                    self.call,
                    node,
                    op,
                    _timeout=_timeout,
                    **{**common, **per_node_args.get(node, {})},
                ): node
                for node in nodes
            }
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    results[node] = future.result()
                except Exception as exc:
                    errors[node] = str(exc)
                    _LOG.error("fanout call failed", node=node, op=op, error=str(exc))

        _LOG.info("fanout", op=op, nodes=len(nodes), ok=len(results), failed=len(errors))
        return results, errors
