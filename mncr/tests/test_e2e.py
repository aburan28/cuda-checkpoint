"""End-to-end flows: checkpoint-and-continue, checkpoint-and-stop, and a
restore onto different nodes with a device map."""

import os
import shutil
import time
import unittest

from . import context  # noqa: F401
from coord.devicemap import build_pairs, to_cli
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


if __name__ == "__main__":
    unittest.main()
