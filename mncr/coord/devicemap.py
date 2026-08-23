"""Build the restore device map.

Two things about --device-map that are easy to get wrong and expensive to
discover late:

  1. It is mandatory on every restore onto hardware that is not the original,
     not just when deliberately migrating. GPU UUIDs are per-device, so a
     restore onto a different node always needs one.

  2. Every GPU visible to CUDA must appear in the map, including devices the job
     never touched. A partial map is rejected.
"""

from mncr.errors import PlacementError

UUID_LEN = len("GPU-00000000-0000-0000-0000-000000000000")


def normalize(uuid):
    text = str(uuid).strip()
    if not text.startswith("GPU-"):
        text = f"GPU-{text}"
    return text


def build_pairs(source_uuids, target_uuids):
    """Positional pairing of source devices onto target devices.

    Positional is the right default: rank N used the Nth visible device, and the
    restored process expects the same ordinal to mean the same thing.
    """
    src = [normalize(u) for u in source_uuids]
    dst = [normalize(u) for u in target_uuids]
    if not src:
        raise PlacementError("source device list is empty")
    if len(src) != len(dst):
        raise PlacementError(
            f"device count mismatch: image was taken with {len(src)} visible "
            f"GPUs, target node exposes {len(dst)}; every visible GPU must be "
            f"mapped"
        )
    return list(zip(src, dst))


def to_cli(pairs):
    """The --device-map argument: oldUuid=newUuid,oldUuid=newUuid,..."""
    return ",".join(f"{old}={new}" for old, new in pairs)


def to_api(pairs):
    """CUcheckpointGpuPair-shaped list, for callers using the driver API."""
    return [{"oldUuid": old, "newUuid": new} for old, new in pairs]


def identity(uuids):
    """Same-hardware restore. Still explicit rather than omitted."""
    normalized = [normalize(u) for u in uuids]
    return list(zip(normalized, normalized))


def is_identity(pairs):
    return all(old == new for old, new in pairs)


def describe(pairs):
    moved = [(o, n) for o, n in pairs if o != n]
    return {
        "devices": len(pairs),
        "moved": len(moved),
        "identity": not moved,
        "cli": to_cli(pairs),
    }
