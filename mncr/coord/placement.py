"""Where an image is allowed to be restored.

Restore constraints are not scheduling preferences - a mismatch is a failed
restore, and a failed restore past the commit point costs the job. So each of
these is a hard filter, and the reason for every rejection is recorded so the
operator can see why a job would not place.
"""

import dataclasses

from mncr.errors import PlacementError
from mncr.version import MIN_DRIVER


@dataclasses.dataclass
class ImageRequirements:
    """Recorded at checkpoint time, enforced at restore time."""

    gpu_count: int
    gpu_model: str
    driver_version: str
    device_memory_mib: int
    host_ram_needed_mib: int
    mnnvl: bool = False

    @staticmethod
    def from_node(node, ranks_on_node=None):
        return ImageRequirements(
            gpu_count=int(node.get("gpu_count", 0)),
            gpu_model=_first_model(node.get("gpu_names", "")),
            driver_version=str(node.get("driver_version", "")),
            device_memory_mib=int(node.get("gpu_mem_total_mib", 0)),
            host_ram_needed_mib=int(node.get("gpu_mem_total_mib", 0)),
            mnnvl=bool(node.get("mnnvl", False)),
        )

    def to_dict(self):
        return dataclasses.asdict(self)


def _first_model(names):
    return (str(names).split("|")[0] or "").strip()


def _driver_major(version):
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return -1


def check(requirements, node, allow_mnnvl=False):
    """Reasons `node` cannot host this image. Empty list means it can."""
    reasons = []

    if int(node.get("gpu_count", 0)) != requirements.gpu_count:
        reasons.append(
            f"gpu count {node.get('gpu_count')} != required {requirements.gpu_count}"
        )

    model = _first_model(node.get("gpu_names", ""))
    if requirements.gpu_model and model != requirements.gpu_model:
        reasons.append(f"gpu model {model!r} != required {requirements.gpu_model!r}")

    target_major = _driver_major(node.get("driver_version"))
    image_major = _driver_major(requirements.driver_version)
    if target_major < MIN_DRIVER:
        reasons.append(f"driver {node.get('driver_version')} below minimum {MIN_DRIVER}")
    elif image_major > 0 and target_major != image_major:
        # Restore across driver major versions is not guaranteed. Refusing is
        # cheaper than discovering it after the commit point.
        reasons.append(
            f"driver major {target_major} != image driver major {image_major}"
        )

    headroom = int(node.get("ram_headroom_mib", 0))
    if headroom < requirements.host_ram_needed_mib:
        reasons.append(
            f"host RAM headroom {headroom} MiB < {requirements.host_ram_needed_mib} "
            f"MiB needed to hold device memory"
        )

    if node.get("mnnvl") and not allow_mnnvl:
        reasons.append("node exposes MNNVL/fabric state; fabric handles cannot be restored")

    if not node.get("criu_cuda_plugin"):
        reasons.append("criu cuda plugin missing")

    return reasons


def select(requirements, nodes, count, allow_mnnvl=False):
    """Choose `count` nodes, or explain why it is impossible."""
    eligible, rejected = [], {}
    for node in nodes:
        reasons = check(requirements, node, allow_mnnvl)
        if reasons:
            rejected[node.get("host", "?")] = reasons
        else:
            eligible.append(node)

    if len(eligible) < count:
        detail = "; ".join(f"{h}: {', '.join(r)}" for h, r in list(rejected.items())[:6])
        raise PlacementError(
            f"need {count} eligible nodes, found {len(eligible)}. {detail}"
        )
    return eligible[:count], rejected
