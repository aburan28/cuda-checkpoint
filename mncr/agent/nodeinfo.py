"""What this node is, in the vocabulary placement and rendezvous use.

The coordinator used to be told about nodes by hand - a dict per node, keyed
the way `coord.placement` expects. On a real cluster that dict is the agent's
to produce: it is the only party standing on the node, and the two fields that
decide a restore, the GPU UUIDs and the address rank 0 can be reached at, are
not things anyone should type.
"""

import os
import shutil
import subprocess

from mncr.netutil import primary_ip as _primary_ip


def _nvidia_smi(query, timeout=30):
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [
        [cell.strip() for cell in row.split(",")]
        for row in proc.stdout.strip().splitlines()
        if row.strip()
    ]


def primary_ip():
    """The address other nodes reach this one at; MNCR_NODE_IP overrides."""
    return _primary_ip("MNCR_NODE_IP")


def mem_available_mib():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0


def describe(node, cfg):
    rows = _nvidia_smi("name,uuid,memory.total,driver_version")
    names = [r[0] for r in rows if len(r) >= 4]
    uuids = [r[1] for r in rows if len(r) >= 4]
    memory = 0
    for row in rows:
        try:
            memory += int(float(row[2]))
        except (ValueError, IndexError):
            pass
    driver = rows[0][3] if rows and len(rows[0]) >= 4 else ""
    return {
        "host": node,
        "hostname": os.uname().nodename,
        "ip": primary_ip(),
        "gpu_count": len(uuids),
        "gpu_names": "|".join(names),
        "gpu_uuids": "|".join(uuids),
        "driver_version": driver,
        "gpu_mem_total_mib": memory,
        "ram_headroom_mib": mem_available_mib(),
        "criu_cuda_plugin": os.path.exists(os.path.join(cfg.criu_libdir, "cuda_plugin.so")),
        "mnnvl": os.environ.get("MNCR_MNNVL", "") not in ("", "0", "false"),
        "cuda_checkpoint": shutil.which(cfg.cuda_checkpoint),
        "criu": shutil.which(cfg.criu),
    }
