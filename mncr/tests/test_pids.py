"""A rank's pid is not a host pid until the agent has made it one."""

import os
import shutil
import sys
import tempfile
import unittest

from . import context  # noqa: F401
from agent.pids import PidResolver
from mncr import rpc


def fake_proc(tmp, table):
    """table: {host_pid: [nspid chain, innermost last]}"""
    root = os.path.join(tmp, "proc")
    for host_pid, chain in table.items():
        pdir = os.path.join(root, str(host_pid))
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, "status"), "w") as fh:
            fh.write(f"Name:\tx\nPid:\t{host_pid}\nNSpid:\t" + "\t".join(str(p) for p in chain) + "\n")
    return root


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_host_namespace_pid_is_itself(self):
        root = fake_proc(self.tmp, {4242: [4242]})
        self.assertEqual(PidResolver(root).resolve(4242), (4242, "host"))

    def test_a_container_pid_is_translated(self):
        # host pid 9001 is pid 7 inside its pod; nothing on the host is pid 7
        root = fake_proc(self.tmp, {9001: [9001, 7], 1: [1]})
        self.assertEqual(PidResolver(root).resolve(7), (9001, "translated"))

    def test_the_same_inner_pid_in_two_pods_is_refused(self):
        root = fake_proc(self.tmp, {9001: [9001, 7], 9002: [9002, 7]})
        with self.assertRaises(LookupError):
            PidResolver(root).resolve(7)

    def test_an_unknown_pid_is_refused(self):
        root = fake_proc(self.tmp, {1: [1]})
        with self.assertRaises(LookupError):
            PidResolver(root).resolve(77)

    def test_a_host_pid_that_is_itself_namespaced_is_not_taken_at_face_value(self):
        # 300 exists on the host but is a pod process; a rank that says "300"
        # means its own inner pid, which is some other host process (or none).
        root = fake_proc(self.tmp, {300: [300, 5], 4000: [4000, 300]})
        self.assertEqual(PidResolver(root).resolve(300), (4000, "translated"))


class PeerPidTest(unittest.TestCase):
    def test_unix_socket_reports_the_peer(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "s.sock")
        server = rpc.Server(f"unix:{path}", name="t")

        @server.op("who")
        def who(peer_pid=None):
            return {"pid": peer_pid}

        @server.op("plain")
        def plain(x):
            return {"x": x}

        server.start()
        try:
            with rpc.Client(f"unix:{path}") as client:
                result = client.call("who")
                # A caller cannot smuggle a pid in: the server's value wins.
                spoofed = client.call("who", peer_pid=1)
                self.assertEqual(client.call("plain", x=3), {"x": 3})
        finally:
            server.stop()
            shutil.rmtree(tmp, ignore_errors=True)
        if sys.platform.startswith("linux"):
            self.assertEqual(result["pid"], os.getpid())
            self.assertEqual(spoofed["pid"], os.getpid())
        else:
            self.assertIsNone(result["pid"])

    def test_tcp_has_no_peer(self):
        server = rpc.Server("tcp:127.0.0.1:0", name="t")

        @server.op("who")
        def who(peer_pid=None):
            return {"pid": peer_pid}

        server.start()
        try:
            with rpc.Client(f"tcp:127.0.0.1:{server.port}") as client:
                self.assertIsNone(client.call("who")["pid"])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
