"""Resource scanning and the two gates it feeds."""

import os
import tempfile
import unittest

from . import context  # noqa: F401
from mncr.procscan import ProcScanner, Severity, summarize


def build_proc(tmp, pid, fds=(), maps=(), tcp=()):
    """tcp: [(inode, local_hex, remote_hex, state_hex)] rows for net/tcp."""
    root = os.path.join(tmp, "proc")
    pdir = os.path.join(root, str(pid))
    os.makedirs(os.path.join(pdir, "fd"), exist_ok=True)
    for index, target in enumerate(fds):
        link = os.path.join(pdir, "fd", str(index + 3))
        real = os.path.join(tmp, f"target{index}")
        open(real, "w").close()
        os.symlink(target, link)
    with open(os.path.join(pdir, "maps"), "w") as fh:
        for path in maps:
            fh.write(f"7f00-7f01 rw-s 00000000 00:06 1234 {path}\n")
    if tcp:
        os.makedirs(os.path.join(pdir, "net"), exist_ok=True)
        with open(os.path.join(pdir, "net", "tcp"), "w") as fh:
            fh.write("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n")
            for index, (inode, local, remote, state) in enumerate(tcp):
                fh.write(
                    f"   {index}: {local} {remote} {state} 00000000:00000000 00:00000000 "
                    f"00000000  1000        0 {inode} 1 0000000000000000 100 0 0 10 0\n"
                )
    return root


class TestProcScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_clean_process_has_no_findings(self):
        root = build_proc(self.tmp, 1, fds=["/tmp/log", "socket:[123]"], maps=["/lib/x.so"])
        self.assertEqual(ProcScanner(root).scan(1), [])

    def test_a_listening_socket_blocks_the_lock(self):
        # 127.0.0.1:28028 listening - NCCL RAS, as measured.
        root = build_proc(
            self.tmp, 7, fds=["socket:[500]", "socket:[501]"],
            tcp=[("500", "0100007F:6D7C", "00000000:0000", "0A")],
        )
        blocking = ProcScanner(root).blocking(7, Severity.BEFORE_LOCK)
        self.assertEqual([f.kind for f in blocking], ["listening_socket"])
        self.assertIn("127.0.0.1:28028", blocking[0].target)
        self.assertIn("NCCL_RAS_ENABLE=0", blocking[0].why)

    def test_an_established_socket_blocks_the_lock_too(self):
        root = build_proc(
            self.tmp, 8, fds=["socket:[600]"],
            tcp=[("600", "CA061FAC:B868", "D9041FAC:1C0D", "01")],
        )
        blocking = ProcScanner(root).blocking(8, Severity.BEFORE_LOCK)
        self.assertEqual([f.kind for f in blocking], ["tcp_socket"])
        self.assertIn("172.31.6.202:47208 -> 172.31.4.217:7181", blocking[0].target)

    def test_non_tcp_sockets_are_not_findings(self):
        # A unix or netlink socket has no net/tcp row; CRIU handles those.
        root = build_proc(
            self.tmp, 9, fds=["socket:[700]"],
            tcp=[("999", "0100007F:6D7C", "00000000:0000", "0A")],
        )
        self.assertEqual(ProcScanner(root).scan(9), [])

    def test_verbs_fd_blocks_the_lock(self):
        root = build_proc(self.tmp, 2, fds=["/dev/infiniband/uverbs3"])
        blocking = ProcScanner(root).blocking(2, Severity.BEFORE_LOCK)
        self.assertEqual([f.kind for f in blocking], ["verbs_fd"])

    def test_nvidia_fd_blocks_only_the_dump(self):
        root = build_proc(self.tmp, 3, fds=["/dev/nvidia0"])
        scanner = ProcScanner(root)
        self.assertEqual(scanner.blocking(3, Severity.BEFORE_LOCK), [])
        self.assertEqual(
            [f.kind for f in scanner.blocking(3, Severity.BEFORE_DUMP)], ["nvidia_fd"]
        )

    def test_dump_gate_includes_lock_gate(self):
        root = build_proc(
            self.tmp, 4, fds=["/dev/infiniband/uverbs0", "/dev/nvidia1"]
        )
        kinds = {f.kind for f in ProcScanner(root).blocking(4, Severity.BEFORE_DUMP)}
        self.assertEqual(kinds, {"verbs_fd", "nvidia_fd"})

    def test_uvm_is_normal_before_the_lock(self):
        """Every CUDA process holds these, measured on real hardware.

        Treating them as a pre-lock blocker rejected a process that checkpoints
        and restores perfectly.
        """
        root = build_proc(
            self.tmp, 5, fds=["/dev/nvidia-uvm"], maps=["/dev/nvidia-uvm"]
        )
        scanner = ProcScanner(root)
        self.assertEqual(scanner.blocking(5, Severity.BEFORE_LOCK), [])
        kinds = {f.kind for f in scanner.blocking(5, Severity.BEFORE_DUMP)}
        self.assertEqual(kinds, {"uvm_fd", "uvm_mapping"})

    def test_an_ordinary_cuda_process_passes_the_lock_gate(self):
        """The shape of a real CUDA process: device fds, control fd, uvm."""
        root = build_proc(
            self.tmp,
            8,
            fds=["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"],
            maps=["/dev/nvidia0", "/dev/nvidia-uvm"],
        )
        self.assertEqual(ProcScanner(root).blocking(8, Severity.BEFORE_LOCK), [])

    def test_gdrcopy_and_summary(self):
        root = build_proc(self.tmp, 6, fds=["/dev/gdrdrv", "/dev/infiniband/rdma_cm"])
        findings = ProcScanner(root).scan(6)
        self.assertEqual(summarize(findings), {"gdrcopy_fd": 1, "verbs_fd": 1})

    def test_missing_pid_is_not_an_error(self):
        root = build_proc(self.tmp, 7)
        self.assertEqual(ProcScanner(root).scan(9999), [])

    def test_unavailable_procfs_reports_unavailable(self):
        self.assertFalse(ProcScanner(os.path.join(self.tmp, "nope")).available())


if __name__ == "__main__":
    unittest.main()
