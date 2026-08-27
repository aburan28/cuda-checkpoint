"""End-to-end flows: checkpoint-and-continue, checkpoint-and-stop, and a
restore onto different nodes with a device map."""

import os
import shutil
import time
import unittest

from . import context  # noqa: F401
from coord.devicemap import build_pairs, is_identity_cli, to_cli
from mncr.proto import Phase
from verify.sim import SimCluster


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimCluster(nodes=2, ranks_per_node=2, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()

    def tearDown(self):
        self.sim.stop()

    def test_repeated_epochs(self):
        for _ in range(3):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        deadline = time.time() + 20
        while time.time() < deadline:
            counts = [
                len([e for e in self.sim.progress(r)["events"] if e["event"] == "resumed"])
                for r in range(4)
            ]
            if all(c >= 3 for c in counts):
                break
            time.sleep(0.05)
        self.assertEqual(counts, [3, 3, 3, 3])

    def test_checkpoint_and_stop_leaves_images(self):
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        self.assertEqual(result["images"], 4)
        epoch = self.sim.coord.store.get(result["epoch_id"])
        self.assertEqual(epoch["phase"], Phase.DUMPED.value)

    def test_restore_onto_different_nodes_builds_device_maps(self):
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        epoch_id = result["epoch_id"]

        # Move every rank to the other node, as a migration would.
        targets = {"node-0": [2, 3], "node-1": [0, 1]}
        restored = self.sim.coord.restore(self.sim.job_id, epoch_id, targets=targets)

        self.assertEqual(restored["phase"], Phase.RUNNING.value)
        maps = restored["device_maps"]
        self.assertEqual(set(maps), {"node-0", "node-1"})
        for node, cli in maps.items():
            pairs = [p.split("=") for p in cli.split(",")]
            self.assertEqual(len(pairs), self.sim.ranks_per_node)
            self.assertTrue(
                all(old != new for old, new in pairs),
                f"{node}: a cross-node restore produced an identity map",
            )

        # Every rank is now owned by the node it moved to.
        for node, agent in self.sim.agents.items():
            owned = sorted(r["rank"] for r in agent.local_ranks(self.sim.job_id))
            self.assertEqual(owned, sorted(targets[node]))

    def test_in_place_restore_still_emits_an_explicit_map(self):
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        restored = self.sim.coord.restore(self.sim.job_id, result["epoch_id"])
        for cli in restored["device_maps"].values():
            pairs = [p.split("=") for p in cli.split(",")]
            self.assertTrue(all(old == new for old, new in pairs))
        # ...but an identity map never reaches criu: on the image's own
        # hardware it says nothing, and can only be wrong.
        for agent in self.sim.agents.values():
            self.assertIsNone(agent.criu.restore_maps[-1])

    def test_migration_hands_criu_the_map_and_finds_manifests_in_the_store(self):
        """A target node that never took the checkpoint has no manifest on
        disk. It must find it in the store, by the source node's name, and it
        must restore with the device map rather than the plugin's default."""
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        epoch_id = result["epoch_id"]
        for node in self.sim.agents:
            stored = os.path.join(
                self.sim.cfg.cache_dir, "shards", epoch_id, f"manifest-{node}.json"
            )
            self.assertTrue(os.path.exists(stored), f"manifest for {node} not in the store")

        # The whole image root, manifests included - exactly what a node that
        # was not there sees.
        shutil.rmtree(os.path.join(self.sim.image_root, epoch_id))

        targets = {"node-0": [2, 3], "node-1": [0, 1]}
        restored = self.sim.coord.restore(self.sim.job_id, epoch_id, targets=targets)
        self.assertEqual(restored["phase"], Phase.RUNNING.value)
        for node, agent in self.sim.agents.items():
            given = agent.criu.restore_maps[-1]
            self.assertEqual(given, restored["device_maps"][node])
            self.assertFalse(is_identity_cli(given), f"{node}: {given}")
            self.assertEqual(agent.driver._device_map, given)

    def test_every_epoch_gets_its_own_rendezvous(self):
        first = self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        second = self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        addresses = [
            self.sim.coord.store.get(e["epoch_id"])["init_method"] for e in (first, second)
        ]
        for addr in addresses:
            self.assertTrue(addr.startswith("tcp://127.0.0.1:"), addr)
        self.assertNotEqual(addresses[0], addresses[1])

    def test_agents_describe_themselves_on_registration(self):
        coord = self.sim.coord
        node, agent = next(iter(self.sim.agents.items()))
        result = coord.register_agent("fresh", agent.rpc_addr)
        info = result["info"]
        self.assertIn("ip", info)
        self.assertIn("gpu_uuids", info)
        self.assertTrue(info["fake"])
        self.assertIs(coord.runner.node_info["fresh"], coord.node_info["fresh"])

    def test_restore_fetches_images_that_are_not_on_the_target_node(self):
        """The case that makes migration real: the image is on another node's
        disk, so the shards have to come back out of the store."""
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        epoch_id = result["epoch_id"]

        epoch_dir = os.path.join(self.sim.image_root, epoch_id)
        manifests = [n for n in os.listdir(epoch_dir) if n.startswith("manifest-")]
        self.assertTrue(manifests, "dump did not write a manifest")

        # Delete the raw CRIU directories, keeping only manifests and shards -
        # exactly what a node that did not take the checkpoint would see.
        for name in os.listdir(epoch_dir):
            path = os.path.join(epoch_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)

        restored = self.sim.coord.restore(
            self.sim.job_id, epoch_id, targets={"node-0": [2, 3], "node-1": [0, 1]}
        )
        self.assertEqual(restored["phase"], Phase.RUNNING.value)
        for rank in range(4):
            self.assertTrue(
                os.path.exists(os.path.join(epoch_dir, f"rank-{rank}", "fake-image.json")),
                f"rank {rank} image was not fetched back from the store",
            )

    def test_pre_dump_pass_runs_before_the_lock(self):
        self.sim.coord.checkpoint(self.sim.job_id, mode="continue", pre_dump=True)
        for node, agent in self.sim.agents.items():
            self.assertTrue(
                agent.criu.dumps, f"{node} recorded no dump"
            )

    def test_device_map_covers_every_visible_gpu(self):
        info = self.sim.coord.node_info["node-0"]
        uuids = info["gpu_uuids"].split("|")
        cli = to_cli(build_pairs(uuids, uuids))
        self.assertEqual(len(cli.split(",")), len(uuids))



class PartialMigrationTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimCluster(nodes=2, ranks_per_node=2, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()

    def tearDown(self):
        self.sim.stop()

    def test_a_node_that_already_holds_ranks_restores_the_newcomers_too(self):
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")
        # rank 2 joins node-0, which already holds 0 and 1
        targets = {"node-0": [0, 1, 2], "node-1": [3]}
        restored = self.sim.coord.restore(self.sim.job_id, result["epoch_id"], targets=targets)
        self.assertEqual(restored["phase"], Phase.RUNNING.value)
        owned = sorted(r["rank"] for r in self.sim.agents["node-0"].local_ranks(self.sim.job_id))
        self.assertEqual(owned, [0, 1, 2])
        self.assertEqual(
            sorted(r["rank"] for r in self.sim.agents["node-1"].local_ranks(self.sim.job_id)), [3]
        )
        self.assertEqual(len(self.sim.agents["node-0"].criu.restores), 3)



class RepeatedRestoreTest(unittest.TestCase):
    """Sources and device maps come from where the image was dumped, not
    from where the last restore put the ranks."""

    def setUp(self):
        self.sim = SimCluster(nodes=2, ranks_per_node=2, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()

    def tearDown(self):
        self.sim.stop()

    def _restore_args(self, epoch_id, **kwargs):
        seen = {}
        real = self.sim.coord.pool.fanout

        def spy(nodes, op, per_node_args=None, **common):
            if op == "restore":
                seen.update(per_node_args or {})
            return real(nodes, op, per_node_args=per_node_args, **common)

        self.sim.coord.pool.fanout = spy
        try:
            result = self.sim.coord.restore(self.sim.job_id, epoch_id, **kwargs)
        finally:
            self.sim.coord.pool.fanout = real
        return result, seen

    def test_a_second_migration_still_fetches_from_the_dumping_node(self):
        epoch_id = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")["epoch_id"]
        dumped = self.sim.coord.store.get(epoch_id)["dumped_on"]
        self.assertEqual(dumped, {"0": "node-0", "1": "node-0", "2": "node-1", "3": "node-1"})

        swap = {"node-0": [2, 3], "node-1": [0, 1]}
        self._restore_args(epoch_id, targets=swap)
        # and back again: the sources must name the original dump nodes
        back = {"node-0": [0, 1], "node-1": [2, 3]}
        result, args = self._restore_args(epoch_id, targets=back)
        self.assertEqual(args["node-0"]["sources"], {0: "node-0", 1: "node-0"})
        self.assertEqual(args["node-1"]["sources"], {2: "node-1", 3: "node-1"})
        # ...and restoring onto the dumping node is an identity map, which
        # the agent then does not hand to criu.
        for node, cli in result["device_maps"].items():
            self.assertTrue(is_identity_cli(cli), f"{node}: {cli}")

    def test_restoring_where_the_ranks_are_after_a_move_maps_from_the_dump(self):
        epoch_id = self.sim.coord.checkpoint(self.sim.job_id, mode="stop")["epoch_id"]
        self._restore_args(epoch_id, targets={"node-0": [2, 3], "node-1": [0, 1]})
        # "where it ran" is now the swapped placement; the image's GPUs are
        # still the original node's, so the map must not be identity.
        result, args = self._restore_args(epoch_id)
        for node, cli in result["device_maps"].items():
            self.assertFalse(is_identity_cli(cli), f"{node}: {cli}")
        self.assertEqual(args["node-0"]["sources"], {2: "node-1", 3: "node-1"})


if __name__ == "__main__":
    unittest.main()
