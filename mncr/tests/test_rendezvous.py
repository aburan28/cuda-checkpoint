"""The rebuild address follows rank 0."""

import unittest

from . import context  # noqa: F401
from coord.rendezvous import Rendezvous
from mncr.proto import RankRef


def _ranks(*placement):
    return [RankRef.make("job", rank, node) for rank, node in placement]


class RendezvousTest(unittest.TestCase):
    def test_address_is_rank_zeros_node(self):
        info = {"a": {"ip": "10.0.0.1"}, "b": {"ip": "10.0.0.2"}}
        rendezvous = Rendezvous(info)
        # rank 0 is listed second and lives on b
        addr = rendezvous.new(_ranks((1, "a"), (0, "b")))
        self.assertTrue(addr.startswith("tcp://10.0.0.2:"), addr)

    def test_ports_rotate(self):
        rendezvous = Rendezvous({"a": {"ip": "10.0.0.1"}}, port_base=1000, port_span=3)
        ports = {int(rendezvous.new(_ranks((0, "a"))).rsplit(":", 1)[1]) for _ in range(3)}
        self.assertEqual(ports, {1000, 1001, 1002})

    def test_none_without_an_address(self):
        rendezvous = Rendezvous({"a": {"host": "a"}})
        self.assertIsNone(rendezvous.new(_ranks((0, "a"))))
        self.assertIsNone(rendezvous.new([]))

    def test_sees_nodes_registered_later(self):
        info = {}
        rendezvous = Rendezvous(info)
        self.assertIsNone(rendezvous.new(_ranks((0, "a"))))
        info["a"] = {"ip": "10.0.0.9"}
        self.assertTrue(rendezvous.new(_ranks((0, "a"))).startswith("tcp://10.0.0.9:"))


if __name__ == "__main__":
    unittest.main()
