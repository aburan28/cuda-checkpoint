"""Shard, compress, checksum, store - and the reverse.

Sizing is the reason this phase exists. The driver copies device memory into
host allocations and CRIU then dumps those as anonymous memory, so a full
8-GPU node produces an image on the order of its total device memory. At that
size the pipeline is the checkpoint's cost, not the driver calls.

Three choices follow from that:

  sharding      fixed-size shards upload in parallel and resume individually
  zstd -1       the cheapest useful ratio; anything slower loses more time in
                CPU than it saves in transfer
  local first   shards land on node NVMe and upload behind the job, so the
                stop-the-world window ends when the dump ends, not when the
                upload does
"""

import concurrent.futures
import os
import shutil
import subprocess
import tarfile
import tempfile

from mncr import log

from .manifest import Manifest, RankImage, Shard, sha256_file

_LOG = log.get("imagestore.pipeline")


def _have(binary):
    return shutil.which(binary) is not None


def pick_codec(preferred="zstd"):
    """zstd if the cluster has it, gzip otherwise. Recorded in the manifest."""
    if preferred == "zstd" and _have("zstd"):
        return "zstd"
    return "gzip"


def _compress(src, dst, codec, level=1):
    if codec == "zstd":
        cmd = ["zstd", f"-{level}", "-q", "-T0", "-f", src, "-o", dst]
    else:
        cmd = ["gzip", "-1", "-c", src]
    if codec == "zstd":
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"zstd failed: {proc.stderr.strip()[:200]}")
    else:
        with open(dst, "wb") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"gzip failed: {proc.stderr.decode()[:200]}")
    return dst


def _decompress(src, dst, codec):
    if codec == "zstd":
        proc = subprocess.run(
            ["zstd", "-d", "-q", "-f", src, "-o", dst], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zstd -d failed: {proc.stderr.strip()[:200]}")
    else:
        with open(dst, "wb") as out:
            proc = subprocess.run(["gzip", "-dc", src], stdout=out, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"gunzip failed: {proc.stderr.decode()[:200]}")
    return dst


def _split(path, shard_bytes, out_dir, prefix):
    """Split one file into fixed-size shards. Returns [(name, path)]."""
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    with open(path, "rb") as fh:
        index = 0
        while True:
            chunk = fh.read(shard_bytes)
            if not chunk:
                break
            name = f"{prefix}.{index:05d}"
            shard_path = os.path.join(out_dir, name)
            with open(shard_path, "wb") as out:
                out.write(chunk)
            shards.append((name, shard_path))
            index += 1
    if not shards:  # an empty image is still an image
        name = f"{prefix}.00000"
        shard_path = os.path.join(out_dir, name)
        open(shard_path, "wb").close()
        shards.append((name, shard_path))
    return shards


class Pipeline:
    def __init__(self, backend, shard_bytes=1 << 30, codec=None, level=1, workers=8,
                 scratch=None):
        self.backend = backend
        self.shard_bytes = int(shard_bytes)
        self.codec = codec or pick_codec()
        self.level = level
        self.workers = workers
        self.scratch = scratch or tempfile.gettempdir()

    # ------------------------------------------------------------------ store
    def store_rank(self, image_id, rank, node, host_pid, images_dir):
        """Pack one rank's CRIU image directory into shards."""
        work = tempfile.mkdtemp(prefix="mncr-pack-", dir=self.scratch)
        try:
            tar_path = os.path.join(work, f"rank-{rank}.tar")
            with tarfile.open(tar_path, "w") as tar:
                tar.add(images_dir, arcname=".")
            raw_bytes = os.path.getsize(tar_path)

            suffix = "zst" if self.codec == "zstd" else "gz"
            comp_path = f"{tar_path}.{suffix}"
            _compress(tar_path, comp_path, self.codec, self.level)
            stored_bytes = os.path.getsize(comp_path)

            pieces = _split(
                comp_path, self.shard_bytes, work, f"rank-{rank}.tar.{suffix}"
            )
            shards = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.workers, len(pieces))
            ) as pool:
                futures = {
                    pool.submit(self._put_shard, image_id, rank, name, path): name
                    for name, path in pieces
                }
                for future in concurrent.futures.as_completed(futures):
                    shards.append(future.result())
            shards.sort(key=lambda s: s.name)

            _LOG.info(
                "rank stored",
                image=image_id,
                rank=rank,
                raw_mib=raw_bytes >> 20,
                stored_mib=stored_bytes >> 20,
                shards=len(shards),
                ratio=round(raw_bytes / max(stored_bytes, 1), 2),
            )
            return RankImage(
                rank=rank,
                node=node,
                host_pid=host_pid,
                shards=shards,
                raw_bytes=raw_bytes,
                stored_bytes=stored_bytes,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _put_shard(self, image_id, rank, name, path):
        key = f"{image_id}/rank-{rank}/{name}"
        digest = sha256_file(path)
        size = os.path.getsize(path)
        self.backend.put(path, key)
        return Shard(name=name, bytes=size, sha256=digest, compressed=True)

    def store_epoch(self, image_id, job_id, epoch_id, entries, requirements=None,
                    source_uuids=None, world_size=0):
        """entries: [{rank, node, host_pid, images_dir}]"""
        manifest = Manifest(
            image_id=image_id,
            job_id=job_id,
            epoch_id=epoch_id,
            world_size=world_size or len(entries),
            requirements=requirements or {},
            source_uuids=source_uuids or {},
            codec=self.codec,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.workers, max(1, len(entries)))
        ) as pool:
            futures = [
                pool.submit(
                    self.store_rank,
                    image_id,
                    entry["rank"],
                    entry["node"],
                    entry.get("host_pid", 0),
                    entry["images_dir"],
                )
                for entry in entries
            ]
            manifest.ranks = [f.result() for f in futures]
        manifest.ranks.sort(key=lambda r: r.rank)
        _LOG.info(
            "epoch stored",
            image=image_id,
            ranks=len(manifest.ranks),
            raw_mib=manifest.raw_bytes >> 20,
            stored_mib=manifest.stored_bytes >> 20,
        )
        return manifest

    # ---------------------------------------------------------------- restore
    def fetch_rank(self, manifest, rank, dest_dir):
        """Reassemble one rank's image directory from its shards."""
        entry = next(
            (
                r
                for r in manifest.ranks
                if (r.rank if isinstance(r, RankImage) else r["rank"]) == rank
            ),
            None,
        )
        if entry is None:
            raise KeyError(f"rank {rank} not in image {manifest.image_id}")
        shards = entry.shards if isinstance(entry, RankImage) else entry["shards"]

        work = tempfile.mkdtemp(prefix="mncr-unpack-", dir=self.scratch)
        try:
            suffix = "zst" if manifest.codec == "zstd" else "gz"
            joined = os.path.join(work, f"rank-{rank}.tar.{suffix}")
            with open(joined, "wb") as out:
                for shard in shards:
                    name = shard.name if isinstance(shard, Shard) else shard["name"]
                    expect = shard.sha256 if isinstance(shard, Shard) else shard["sha256"]
                    key = f"{manifest.image_id}/rank-{rank}/{name}"
                    local = os.path.join(work, name)
                    self.backend.get(key, local)
                    actual = sha256_file(local)
                    if actual != expect:
                        raise RuntimeError(
                            f"shard {name} checksum mismatch: image is corrupt"
                        )
                    with open(local, "rb") as fh:
                        shutil.copyfileobj(fh, out)
                    os.unlink(local)

            tar_path = os.path.join(work, f"rank-{rank}.tar")
            _decompress(joined, tar_path, manifest.codec)
            os.makedirs(dest_dir, exist_ok=True)
            with tarfile.open(tar_path) as tar:
                tar.extractall(dest_dir)
            _LOG.info("rank fetched", image=manifest.image_id, rank=rank, dest=dest_dir)
            return dest_dir
        finally:
            shutil.rmtree(work, ignore_errors=True)
