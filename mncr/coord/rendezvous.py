"""A fresh rendezvous address for every rebuild.

The process group is destroyed before the checkpoint and rebuilt after it, and
the rebuild needs somewhere to meet. Reusing the launcher's MASTER_ADDR works
exactly as long as nothing moves: after a migration rank 0 is on a different
node, and every peer would wait on an address nobody is listening at.

So the coordinator issues the address. It is the only party that knows where
rank 0 is *now* - the placement it just decided - and it hands the same string
to every rank in the token. The port rotates so two epochs in quick succession
never contend for a socket still in TIME_WAIT.
"""

import time

DEFAULT_PORT_BASE = 29600
DEFAULT_PORT_SPAN = 300


class Rendezvous:
    def __init__(self, node_info, port_base=DEFAULT_PORT_BASE, port_span=DEFAULT_PORT_SPAN):
        """node_info is shared with the coordinator and read at call time, so
        nodes registered after construction are seen."""
        self.node_info = node_info
        self.port_base = int(port_base)
        self.port_span = max(1, int(port_span))
        # Seeded from the clock so a restarted coordinator does not start its
        # rotation on a port the previous incarnation just used.
        self._seq = int(time.time()) % self.port_span

    def host_for(self, node):
        info = self.node_info.get(node) or {}
        return info.get("ip") or info.get("addr") or None

    def new(self, ranks):
        """tcp://<rank-0's node>:<port>, or None if that node has no known address.

        None means "use env://", which is right on a single node that never
        registered an address and wrong everywhere else; the rebuild logs it.
        """
        if not ranks:
            return None
        lowest = min(ranks, key=lambda r: int(r["rank"]))
        host = self.host_for(lowest["node"])
        if not host:
            return None
        port = self.port_base + (self._seq % self.port_span)
        self._seq += 1
        return f"tcp://{host}:{port}"
