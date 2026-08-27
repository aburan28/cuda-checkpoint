"""Restore placement. Every rule here is a failed restore if it is not enforced."""

import unittest

from . import context  # noqa: F401
from coord.placement import ImageRequirements, check, select
from mncr.errors import PlacementError


def node(host="n", **over):
    base = {
        "host": host,
        "gpu_count": 8,
        "gpu_names": "H100|H100|H100|H100|H100|H100|H100|H100",
        "driver_version": "610.57.04",
        "gpu_mem_total_mib": 655360,
        "ram_headroom_mib": 900000,
        "criu_cuda_plugin": True,
        "mnnvl": False,
    }
    base.update(over)
    return base


REQ = ImageRequirements(
    gpu_count=8,
    gpu_model="H100",
    driver_version="610.57.04",
    device_memory_mib=655360,
    host_ram_needed_mib=655360,
)


class TestPlacement(unittest.TestCase):
    def test_matching_node_accepted(self):
        self.assertEqual(check(REQ, node()), [])

    def test_gpu_count_mismatch(self):
        self.assertIn("gpu count", check(REQ, node(gpu_count=4))[0])

    def test_gpu_model_mismatch(self):
        reasons = check(REQ, node(gpu_names="A100|A100"))
        self.assertTrue(any("gpu model" in r for r in reasons))

    def test_driver_major_mismatch_rejected(self):
        reasons = check(REQ, node(driver_version="620.11"))
        self.assertTrue(any("image driver major" in r for r in reasons))

    def test_driver_below_minimum_rejected(self):
        reasons = check(REQ, node(driver_version="580.1"))
        self.assertTrue(any("below minimum" in r for r in reasons))

    def test_insufficient_host_ram_rejected(self):
        reasons = check(REQ, node(ram_headroom_mib=1000))
        self.assertTrue(any("host RAM headroom" in r for r in reasons))

    def test_mnnvl_rejected_by_default_allowed_when_forced(self):
        self.assertTrue(check(REQ, node(mnnvl=True)))
        self.assertEqual(check(REQ, node(mnnvl=True), allow_mnnvl=True), [])

    def test_missing_criu_plugin_rejected(self):
        self.assertTrue(check(REQ, node(criu_cuda_plugin=False)))

    def test_select_explains_why_it_cannot_place(self):
        with self.assertRaises(PlacementError) as ctx:
            select(REQ, [node("bad", mnnvl=True)], 1)
        self.assertIn("bad", str(ctx.exception))
        self.assertIn("fabric", str(ctx.exception))

    def test_select_returns_requested_count(self):
        chosen, rejected = select(REQ, [node("a"), node("b"), node("c", gpu_count=2)], 2)
        self.assertEqual([n["host"] for n in chosen], ["a", "b"])
        self.assertIn("c", rejected)


if __name__ == "__main__":
    unittest.main()
