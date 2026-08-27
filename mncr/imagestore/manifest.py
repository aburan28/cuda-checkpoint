"""Image manifest: what an image is, and what it needs to come back.

Two fields carry most of the weight. `requirements` is what placement enforces
at restore time - GPU count, model, driver major, host RAM. `source_uuids` is
what the device map is built from; without it a cross-node restore has no way to
construct a complete map, and a partial map is rejected by the driver.
"""

import dataclasses
import hashlib
import json
import os
import time


@dataclasses.dataclass
class Shard:
    name: str
    bytes: int
    sha256: str
    compressed: bool

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RankImage:
    rank: int
    node: str
    host_pid: int
    shards: list
    raw_bytes: int = 0
    stored_bytes: int = 0

    def to_dict(self):
        return {
            "rank": self.rank,
            "node": self.node,
            "host_pid": self.host_pid,
            "raw_bytes": self.raw_bytes,
            "stored_bytes": self.stored_bytes,
            "shards": [s.to_dict() if isinstance(s, Shard) else s for s in self.shards],
        }


@dataclasses.dataclass
class Manifest:
    image_id: str
    job_id: str
    epoch_id: str
    created_at: float = dataclasses.field(default_factory=time.time)
    world_size: int = 0
    ranks: list = dataclasses.field(default_factory=list)
    requirements: dict = dataclasses.field(default_factory=dict)
    source_uuids: dict = dataclasses.field(default_factory=dict)  # node -> [uuid]
    codec: str = "zstd"
    version: int = 1

    def to_dict(self):
        return {
            "version": self.version,
            "image_id": self.image_id,
            "job_id": self.job_id,
            "epoch_id": self.epoch_id,
            "created_at": self.created_at,
            "world_size": self.world_size,
            "codec": self.codec,
            "requirements": self.requirements,
            "source_uuids": self.source_uuids,
            "ranks": [r.to_dict() if isinstance(r, RankImage) else r for r in self.ranks],
        }

    @property
    def raw_bytes(self):
        return sum(
            (r.raw_bytes if isinstance(r, RankImage) else r.get("raw_bytes", 0))
            for r in self.ranks
        )

    @property
    def stored_bytes(self):
        return sum(
            (r.stored_bytes if isinstance(r, RankImage) else r.get("stored_bytes", 0))
            for r in self.ranks
        )

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @staticmethod
    def load(path):
        with open(path) as fh:
            data = json.load(fh)
        return Manifest(
            image_id=data["image_id"],
            job_id=data["job_id"],
            epoch_id=data["epoch_id"],
            created_at=data.get("created_at", 0),
            world_size=data.get("world_size", 0),
            ranks=data.get("ranks", []),
            requirements=data.get("requirements", {}),
            source_uuids=data.get("source_uuids", {}),
            codec=data.get("codec", "zstd"),
            version=data.get("version", 1),
        )


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
