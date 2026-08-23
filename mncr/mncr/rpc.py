"""JSON-line RPC over AF_UNIX or AF_INET, stdlib only.

Deliberately dependency-free: the agent runs as a privileged DaemonSet and the
rank library is injected into somebody else's training image. Neither can afford
to drag in a transport stack.

Address forms:
    "unix:/run/mncr/agent.sock"   filesystem socket
    "unix:@mncr-agent"            Linux abstract socket (no filesystem entry)
    "tcp:0.0.0.0:7181"            TCP
"""

import inspect
import socket
import socketserver
import struct
import threading

from . import log
from .proto import Message

_LOG = log.get("rpc")
_MAX_LINE = 8 << 20  # a device map for 8 GPUs is ~700 bytes; 8 MiB is slack


def parse_addr(addr):
    if addr.startswith("unix:"):
        path = addr[5:]
        if path.startswith("@"):
            # Abstract namespace: leading NUL, Linux only.
            return socket.AF_UNIX, "\0" + path[1:]
        return socket.AF_UNIX, path
    if addr.startswith("tcp:"):
        host, _, port = addr[4:].rpartition(":")
        return socket.AF_INET, (host or "0.0.0.0", int(port))
    raise ValueError(f"unsupported address {addr!r}")


def peer_pid(sock):
    """The connecting process's pid, as this process sees it. AF_UNIX only.

    This is how the agent learns a rank's host pid without trusting the rank:
    a process in a pod reports the pid it has inside the pod, which means
    nothing to the driver, but SO_PEERCRED is translated by the kernel into
    the receiver's pid namespace - the host's, since the agent runs there.
    None over TCP, or on a platform without SO_PEERCRED.
    """
    if sock.family != socket.AF_UNIX or not hasattr(socket, "SO_PEERCRED"):
        return None
    try:
        creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        pid, _uid, _gid = struct.unpack("3i", creds)
    except (OSError, struct.error):
        return None
    return pid or None


def accepts_peer_pid(fn):
    try:
        return "peer_pid" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        pid = peer_pid(self.request)
        while True:
            line = self.rfile.readline(_MAX_LINE)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                req = Message.decode(line)
            except Exception as exc:
                self.wfile.write(Message.err("?", f"malformed request: {exc}").encode())
                self.wfile.flush()
                return
            resp = self.server.dispatch(req, peer_pid=pid)
            self.wfile.write(resp.encode())
            self.wfile.flush()


class _ThreadedUnix(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class _ThreadedTcp(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Server:
    """Registers handlers by op name and serves them on a background thread."""

    def __init__(self, addr, name="server"):
        self.addr = addr
        self.name = name
        self._ops = {}
        self._wants_peer = {}
        self._srv = None
        self._thread = None
        family, target = parse_addr(addr)
        cls = _ThreadedUnix if family == socket.AF_UNIX else _ThreadedTcp
        if family == socket.AF_UNIX and isinstance(target, str) and not target.startswith("\0"):
            import os

            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
        self._srv = cls(target, _Handler)
        self._srv.dispatch = self._dispatch

    def op(self, name):
        def deco(fn):
            self._ops[name] = fn
            self._wants_peer[name] = accepts_peer_pid(fn)
            return fn

        return deco

    def _dispatch(self, req, peer_pid=None):
        name = req.get("op")
        fn = self._ops.get(name)
        if fn is None:
            return Message.err(req.get("id", "?"), f"unknown op {name!r}")
        args = dict(req.get("args", {}))
        if peer_pid is not None and self._wants_peer.get(name):
            # Handed to handlers that ask for it; never something the caller
            # can set, since it overrides whatever they sent.
            args["peer_pid"] = peer_pid
        try:
            result = fn(**args)
            return Message.ok(req["id"], **(result or {}))
        except Exception as exc:  # handlers convert their own domain errors
            _LOG.warn("op failed", server=self.name, op=req.get("op"), error=str(exc))
            return Message.err(req["id"], exc, kind=type(exc).__name__)

    @property
    def port(self):
        """Bound port, for tests that ask for :0."""
        return self._srv.server_address[1]

    def start(self):
        self._thread = threading.Thread(
            target=self._srv.serve_forever, name=f"{self.name}-rpc", daemon=True
        )
        self._thread.start()
        _LOG.info("listening", server=self.name, addr=self.addr)
        return self

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
        if self._thread:
            self._thread.join(timeout=5)


class Client:
    """One connection, one call at a time. Callers hold one per peer."""

    def __init__(self, addr, timeout=30.0):
        self.addr = addr
        self.timeout = timeout
        self._sock = None
        self._rfile = None

    def connect(self):
        family, target = parse_addr(self.addr)
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(target)
        except BaseException:
            # A refused connection is routine here - an agent restarting, a
            # coordinator that has not come up yet - and the caller's `with`
            # block never runs its exit, so the socket has to be closed on the
            # way out or it leaks on every retry.
            sock.close()
            raise
        self._sock = sock
        self._rfile = sock.makefile("rb")
        return self

    def call(self, op, _timeout=None, **args):
        if self._sock is None:
            self.connect()
        if _timeout is not None:
            self._sock.settimeout(_timeout)
        req = Message.request(op, **args)
        self._sock.sendall(req.encode())
        line = self._rfile.readline(_MAX_LINE)
        if _timeout is not None:
            self._sock.settimeout(self.timeout)
        if not line:
            raise ConnectionError(f"{self.addr}: peer closed during {op}")
        resp = Message.decode(line)
        if not resp.get("ok"):
            raise RemoteError(resp.get("error", "unknown"), resp.get("kind", "error"), op)
        return resp.get("result", {})

    def close(self):
        for handle in (self._rfile, self._sock):
            try:
                if handle:
                    handle.close()
            except OSError:
                pass
        self._rfile = self._sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()


class RemoteError(Exception):
    """The peer handled the call and returned failure."""

    def __init__(self, message, kind, op):
        super().__init__(f"{op}: {message}")
        self.kind = kind
        self.op = op
