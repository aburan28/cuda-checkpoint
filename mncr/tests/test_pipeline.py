"""Image pipeline: round trip, corruption detection, sharding, cache policy."""

import os
import shutil
import tempfile
import unittest

from . import context  # noqa: F401
from imagestore.backends import LocalBackend, TieredBackend
from imagestore.cache import ImageCache
from imagestore.manifest import Manifest
from imagestore.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "store")
        self.pipe = Pipeline(
            LocalBackend(self.store), shard_bytes=64 << 10, workers=4, scratch=self.tmp
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _image_dir(self, rank, payload=None):
        path = os.path.join(self.tmp, "criu", f"rank-{rank}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "pages-1.img"), "wb") as fh:
            fh.write(payload if payload is not None else os.urandom(300 << 10))
        return path

    def test_round_trip_is_byte_identical(self):
        payload = os.urandom(200 << 10)
        src = self._image_dir(0, payload)
        manifest = self.pipe.store_epoch(
            "img", "job", "ep", [{"rank": 0, "node": "n", "host_pid": 5, "images_dir": src}]
        )
        dest = self.pipe.fetch_rank(manifest, 0, os.path.join(self.tmp, "out"))
        with open(os.path.join(dest, "pages-1.img"), "rb") as fh:
            self.assertEqual(fh.read(), payload)

    def test_large_image_is_sharded(self):
        src = self._image_dir(0, os.urandom(500 << 10))
        manifest = self.pipe.store_epoch(
            "img", "job", "ep", [{"rank": 0, "node": "n", "host_pid": 5, "images_dir": src}]
        )
        self.assertGreater(len(manifest.ranks[0].shards), 1)

    def test_corrupt_shard_is_detected(self):
        src = self._image_dir(0)
        manifest = self.pipe.store_epoch(
            "img", "job", "ep", [{"rank": 0, "node": "n", "host_pid": 5, "images_dir": src}]
        )
        shard = manifest.ranks[0].shards[0]
        target = os.path.join(self.store, "img", "rank-0", shard.name)
        with open(target, "r+b") as fh:
            fh.seek(0)
            fh.write(b"\xff\xff\xff\xff")
        with self.assertRaises(RuntimeError) as ctx:
            self.pipe.fetch_rank(manifest, 0, os.path.join(self.tmp, "out"))
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_manifest_survives_save_and_load(self):
        src = self._image_dir(0)
        manifest = self.pipe.store_epoch(
            "img",
            "job",
            "ep",
            [{"rank": 0, "node": "n", "host_pid": 5, "images_dir": src}],
            requirements={"gpu_count": 8},
            source_uuids={"n": ["GPU-a"]},
        )
        path = manifest.save(os.path.join(self.tmp, "m.json"))
        loaded = Manifest.load(path)
        self.assertEqual(loaded.requirements["gpu_count"], 8)
        self.assertEqual(loaded.source_uuids["n"], ["GPU-a"])
        dest = self.pipe.fetch_rank(loaded, 0, os.path.join(self.tmp, "out2"))
        self.assertTrue(os.path.exists(os.path.join(dest, "pages-1.img")))

    def test_multiple_ranks_stored_in_parallel(self):
        entries = [
            {"rank": r, "node": "n", "host_pid": 100 + r, "images_dir": self._image_dir(r)}
            for r in range(4)
        ]
        manifest = self.pipe.store_epoch("img", "job", "ep", entries)
        self.assertEqual([r.rank for r in manifest.ranks], [0, 1, 2, 3])
        self.assertGreater(manifest.raw_bytes, 0)

    def test_unknown_rank_raises(self):
        src = self._image_dir(0)
        manifest = self.pipe.store_epoch(
            "img", "job", "ep", [{"rank": 0, "node": "n", "host_pid": 5, "images_dir": src}]
        )
        with self.assertRaises(KeyError):
            self.pipe.fetch_rank(manifest, 9, os.path.join(self.tmp, "out"))


class TestTieredBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_local_hit_avoids_remote(self):
        local = LocalBackend(os.path.join(self.tmp, "local"))
        remote = LocalBackend(os.path.join(self.tmp, "remote"))
        tiered = TieredBackend(local, remote)
        src = os.path.join(self.tmp, "f")
        with open(src, "wb") as fh:
            fh.write(b"payload")
        tiered.put(src, "k")
        self.assertTrue(local.exists("k"))
        self.assertTrue(remote.exists("k"))
        remote.delete("k")           # prove the read came from local
        out = tiered.get("k", os.path.join(self.tmp, "out"))
        self.assertEqual(open(out, "rb").read(), b"payload")


class TestImageCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _image(self, cache, name, size):
        path = os.path.join(cache.root, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "blob"), "wb") as fh:
            fh.write(b"\0" * size)
        cache.mark_complete(name)
        return path

    def test_incomplete_image_is_not_a_hit(self):
        cache = ImageCache(self.tmp)
        os.makedirs(os.path.join(self.tmp, "half"), exist_ok=True)
        self.assertFalse(cache.has("half"))

    def test_evicts_whole_images_oldest_first(self):
        cache = ImageCache(self.tmp, capacity_bytes=1500)
        for index, name in enumerate(("old", "mid", "new")):
            self._image(cache, name, 1000)
            os.utime(
                os.path.join(cache.root, name, ".complete"), (index, index)
            )
        evicted = cache.enforce()
        self.assertIn("old", evicted)
        self.assertTrue(cache.has("new"))

    def test_pinned_image_is_never_evicted(self):
        cache = ImageCache(self.tmp, capacity_bytes=500)
        for name in ("a", "b"):
            self._image(cache, name, 1000)
        evicted = cache.enforce(keep={"a"})
        self.assertNotIn("a", evicted)



class TestSplit(unittest.TestCase):
    """Splitting streams; it never holds a shard in memory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _split(self, payload, shard_bytes, block):
        from imagestore.pipeline import _split

        src = os.path.join(self.tmp, "in")
        with open(src, "wb") as fh:
            fh.write(payload)
        shards = _split(src, shard_bytes, os.path.join(self.tmp, "out"), "p", block=block)
        joined = b"".join(open(path, "rb").read() for _, path in shards)
        return shards, joined

    def test_blocks_smaller_than_shards_reassemble(self):
        payload = os.urandom(10_000)
        shards, joined = self._split(payload, shard_bytes=4096, block=1000)
        self.assertEqual([os.path.getsize(p) for _, p in shards], [4096, 4096, 1808])
        self.assertEqual(joined, payload)

    def test_no_empty_trailing_shard_on_a_boundary(self):
        payload = os.urandom(8192)
        shards, joined = self._split(payload, shard_bytes=4096, block=4096)
        self.assertEqual(len(shards), 2)
        self.assertEqual(joined, payload)
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "out"))), ["p.00000", "p.00001"])

    def test_an_empty_input_is_one_empty_shard(self):
        shards, joined = self._split(b"", shard_bytes=4096, block=64)
        self.assertEqual(len(shards), 1)
        self.assertEqual(joined, b"")



class TestRemoteDelete(unittest.TestCase):
    def test_tiered_delete_reaches_both_tiers(self):
        tmp = tempfile.mkdtemp()
        try:
            local = LocalBackend(os.path.join(tmp, "local"))
            src = os.path.join(tmp, "blob")
            open(src, "wb").write(b"x")
            local.put(src, "ep/rank-0/s.00000")

            class Remote:
                def __init__(self):
                    self.deleted = []

                def delete(self, key):
                    self.deleted.append(key)
                    return True

                def exists(self, key):
                    return False

            remote = Remote()
            tiered = TieredBackend(local, remote)
            self.assertTrue(tiered.delete("ep/rank-0/s.00000"))
            self.assertEqual(remote.deleted, ["ep/rank-0/s.00000"])
            self.assertFalse(local.exists("ep/rank-0/s.00000"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_remote_backend_runs_the_delete_template(self):
        from imagestore.backends import RemoteBackend

        ok = RemoteBackend("s3://bucket/prefix", delete_template="test -n {dst}")
        self.assertTrue(ok.delete("ep/manifest.json"))
        failing = RemoteBackend("s3://bucket/prefix", delete_template="false {dst}")
        with self.assertRaises(RuntimeError):
            failing.delete("ep/manifest.json")


if __name__ == "__main__":
    unittest.main()
