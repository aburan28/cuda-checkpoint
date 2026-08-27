"""Configuration, resolved from environment with documented defaults.

Every knob that can move a production decision lives here rather than being
spelled inline, so `python3 -m mncr.config` prints the effective settings on a
node when something behaves unexpectedly.
"""

import dataclasses
import json
import os


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.lower() not in ("0", "false", "no")
    return cast(raw)


@dataclasses.dataclass
class Config:
    # transports
    agent_addr: str = _env("MNCR_AGENT_ADDR", "tcp:0.0.0.0:7181")
    rank_addr: str = _env("MNCR_RANK_ADDR", "unix:@mncr-agent")
    coord_addr: str = _env("MNCR_COORD_ADDR", "tcp:0.0.0.0:7180")

    # timeouts, seconds
    safe_point_timeout: float = _env("MNCR_SAFE_POINT_TIMEOUT", 300.0, float)
    quiesce_timeout: float = _env("MNCR_QUIESCE_TIMEOUT", 120.0, float)
    lock_timeout_ms: int = _env("MNCR_LOCK_TIMEOUT_MS", 60000, int)
    checkpoint_timeout: float = _env("MNCR_CHECKPOINT_TIMEOUT", 900.0, float)
    dump_timeout: float = _env("MNCR_DUMP_TIMEOUT", 3600.0, float)
    # How long a rank waits for its own copy of a request its peers already
    # hold before acting on what the collective carried.
    request_settle: float = _env("MNCR_REQUEST_SETTLE", 5.0, float)

    # paths
    jobfile_dir: str = _env("MNCR_JOBFILE_DIR", "/run/mncr/jobs")
    image_dir: str = _env("MNCR_IMAGE_DIR", "/var/lib/mncr/images")
    cache_dir: str = _env("MNCR_CACHE_DIR", "/var/lib/mncr/cache")

    # binaries
    cuda_checkpoint: str = _env("MNCR_CUDA_CHECKPOINT", "cuda-checkpoint")
    criu: str = _env("MNCR_CRIU", "criu")
    criu_libdir: str = _env("MNCR_CRIU_LIBDIR", "/usr/lib/criu")

    # behaviour
    fake: bool = _env("MNCR_FAKE", False, bool)
    # Where the agent's verification gates look. Injectable so a fake-driver
    # run on a GPU node does not fail its dump gate on fds the fake driver
    # never released.
    proc_root: str = _env("MNCR_PROC_ROOT", "/proc")
    fake_call_latency: float = _env("MNCR_FAKE_CALL_LATENCY", 0.0, float)
    strict_clean: bool = _env("MNCR_STRICT_CLEAN", True, bool)
    allow_mnnvl: bool = _env("MNCR_ALLOW_MNNVL", False, bool)
    shard_bytes: int = _env("MNCR_SHARD_BYTES", 1 << 30, int)
    zstd_level: int = _env("MNCR_ZSTD_LEVEL", 1, int)

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


def load():
    return Config()


if __name__ == "__main__":
    print(load().to_json())
