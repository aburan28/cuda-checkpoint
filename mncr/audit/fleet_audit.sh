#!/usr/bin/env bash
# fleet_audit.sh - node inventory for P0.
#
# Emits one JSON object on stdout describing everything that decides whether
# this node can participate in checkpoint/restore. Run it as a DaemonSet job or
# over ssh; feed the collected objects to report.py.
#
# Exit status is always 0 - a node that cannot participate is a finding, not an
# error. report.py decides go/no-go.

set -uo pipefail

j_str() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/ /g'; }
have()  { command -v "$1" >/dev/null 2>&1; }

hostname_val=$(hostname 2>/dev/null || echo unknown)
kernel_val=$(uname -r 2>/dev/null || echo unknown)

# ------------------------------------------------------------------ driver
driver_version="none"
gpu_count=0
gpu_names=""
gpu_uuids=""
gpu_mem_total_mib=0
persistence="unknown"
if have nvidia-smi; then
  driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$driver_version" ] && driver_version="none"
  while IFS=, read -r name uuid mem pmode; do
    [ -z "$name" ] && continue
    gpu_count=$((gpu_count + 1))
    gpu_names="${gpu_names}${gpu_names:+|}$(printf '%s' "$name" | tr -d ' ' )"
    gpu_uuids="${gpu_uuids}${gpu_uuids:+|}$(printf '%s' "$uuid" | tr -d ' ')"
    mem_num=$(printf '%s' "$mem" | tr -dc '0-9')
    gpu_mem_total_mib=$((gpu_mem_total_mib + ${mem_num:-0}))
    persistence=$(printf '%s' "$pmode" | tr -d ' ')
  done < <(nvidia-smi --query-gpu=name,uuid,memory.total,persistence_mode --format=csv,noheader 2>/dev/null)
fi

# MNNVL / IMEX. Present as Fabric state in nvidia-smi -q on NVL systems.
mnnvl="false"
if have nvidia-smi && nvidia-smi -q 2>/dev/null | grep -qiE '^[[:space:]]*(Fabric|ClusterUUID|CliqueId)'; then
  mnnvl="true"
fi

# --------------------------------------------------------------------- host
mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
host_ram_mib=$(( ${mem_total_kib:-0} / 1024 ))
# The sizing constraint: device memory is copied into host allocations, so the
# node needs at least that much RAM free on top of what the job already uses.
ram_headroom_mib=$(( host_ram_mib - gpu_mem_total_mib ))

# --------------------------------------------------------------------- criu
criu_version="none"
criu_plugin="false"
if have criu; then
  criu_version=$(criu --version 2>/dev/null | head -1 | awk '{print $2}')
  [ -z "$criu_version" ] && criu_version="unknown"
fi
for d in /usr/lib/criu /usr/local/lib/criu /usr/lib64/criu; do
  [ -f "$d/cuda_plugin.so" ] && criu_plugin="true" && break
done

# ------------------------------------------------------------------- rdma
rdma_devices=0
if [ -d /sys/class/infiniband ]; then
  rdma_devices=$(find /sys/class/infiniband -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
fi
peermem="false"
if [ -f /proc/modules ] && grep -q '^nvidia_peermem' /proc/modules 2>/dev/null; then
  peermem="true"
fi

# ------------------------------------------------------- cuda-checkpoint
cc_path="none"
if have cuda-checkpoint; then
  cc_path=$(command -v cuda-checkpoint)
fi

# ------------------------------------------------------------------ output
cat <<JSON
{
  "host": "$(j_str "$hostname_val")",
  "kernel": "$(j_str "$kernel_val")",
  "driver_version": "$(j_str "$driver_version")",
  "gpu_count": ${gpu_count},
  "gpu_names": "$(j_str "$gpu_names")",
  "gpu_uuids": "$(j_str "$gpu_uuids")",
  "gpu_mem_total_mib": ${gpu_mem_total_mib},
  "persistence_mode": "$(j_str "$persistence")",
  "mnnvl": ${mnnvl},
  "host_ram_mib": ${host_ram_mib},
  "ram_headroom_mib": ${ram_headroom_mib},
  "criu_version": "$(j_str "$criu_version")",
  "criu_cuda_plugin": ${criu_plugin},
  "rdma_devices": ${rdma_devices},
  "nvidia_peermem": ${peermem},
  "cuda_checkpoint": "$(j_str "$cc_path")"
}
JSON
