"""Registry of CUDA graphs that must be recaptured after a restore.

A captured graph survives the checkpoint as memory. The communicator baked into
it does not. Any graph containing a collective is invalid the moment the process
group is destroyed, so the rank has to know which graphs those are and how to
rebuild them.

This matters most for the cold-start use case, where graph capture is a large
share of the startup cost being skipped.
"""

from mncr import log

_LOG = log.get("torchckpt.graphs")


class GraphRegistry:
    def __init__(self):
        self._entries = []

    def register(self, name, recapture, contains_collective=True):
        """Register a recapture callable.

        `recapture()` must rebuild the graph and rebind whatever holds it. It is
        called after the process group has been re-initialised.
        """
        self._entries.append(
            {
                "name": name,
                "recapture": recapture,
                "contains_collective": bool(contains_collective),
            }
        )
        return self

    def clear(self):
        self._entries.clear()

    def names(self):
        return [e["name"] for e in self._entries]

    def invalidated_by_teardown(self):
        return [e["name"] for e in self._entries if e["contains_collective"]]

    def recapture_all(self):
        done, failed = [], []
        for entry in self._entries:
            try:
                entry["recapture"]()
                done.append(entry["name"])
            except Exception as exc:
                _LOG.error("recapture failed", graph=entry["name"], error=str(exc))
                failed.append({"graph": entry["name"], "error": str(exc)})
        _LOG.info("graphs recaptured", count=len(done), failed=len(failed))
        if failed:
            raise RuntimeError(f"graph recapture failed: {failed}")
        return done


REGISTRY = GraphRegistry()


def register(name, recapture, contains_collective=True):
    """Module-level convenience: torchckpt.graphs.register("decode", fn)."""
    return REGISTRY.register(name, recapture, contains_collective)
