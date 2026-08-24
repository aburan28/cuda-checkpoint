#!/usr/bin/env bash
# bootstrap-node.sh - make a GPU node able to checkpoint.
#
# Written after a spot node was reclaimed mid-session and took a hand-built
# CRIU with it. Everything here is idempotent: re-running on a provisioned node
# does nothing and says so, which is what makes it safe to put in cloud-init or
# a DaemonSet init container.
#
#   sudo ./bootstrap-node.sh --cuda-checkpoint ./cuda-checkpoint
#   sudo ./bootstrap-node.sh --criu-version v4.2.1 --skip-preflight
#
# What it installs:
#   cuda-checkpoint   from the path given, or from PATH if already present
#   criu              built from source, because no distro ships 4.x with the
#                     CUDA plugin, and the plugin is the whole point
#   cuda_plugin.so    into /usr/lib/criu, which is where criu looks
#
# Exit status is preflight's: 0 if the node can participate, 1 if it cannot.

set -uo pipefail

CRIU_VERSION="v4.2.1"
CRIU_MIN_MAJOR=4
CUDA_CHECKPOINT_SRC=""
PLUGIN_DIR="/usr/lib/criu"
BUILD_DIR="/opt/criu-src"
SKIP_PREFLIGHT=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda-checkpoint) CUDA_CHECKPOINT_SRC="$2"; shift 2 ;;
    --criu-version)    CRIU_VERSION="$2"; shift 2 ;;
    --plugin-dir)      PLUGIN_DIR="$2"; shift 2 ;;
    --build-dir)       BUILD_DIR="$2"; shift 2 ;;
    --skip-preflight)  SKIP_PREFLIGHT=1; shift ;;
    -h|--help)         usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n!!  %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run me as root: CRIU and the driver calls both need it"

# ------------------------------------------------------------------ driver
say "GPU and driver"
if ! command -v nvidia-smi >/dev/null; then
  die "no nvidia-smi; this is not a GPU node"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read -r line; do
  note "$line"
done

# ------------------------------------------------------- cuda-checkpoint
say "cuda-checkpoint"
if [ -n "$CUDA_CHECKPOINT_SRC" ]; then
  [ -f "$CUDA_CHECKPOINT_SRC" ] || die "not found: $CUDA_CHECKPOINT_SRC"
  install -m0755 "$CUDA_CHECKPOINT_SRC" /usr/local/bin/cuda-checkpoint
  note "installed from $CUDA_CHECKPOINT_SRC"
elif command -v cuda-checkpoint >/dev/null; then
  note "already present at $(command -v cuda-checkpoint)"
else
  die "cuda-checkpoint not found. Pass --cuda-checkpoint <path>; the binary is
    in bin/x86_64_Linux or bin/aarch64_Linux of the NVIDIA/cuda-checkpoint repo."
fi
# The utility is a shim that reports the driver's version, not its own.
note "$(cuda-checkpoint --help 2>&1 | sed -n 2p)"

# --------------------------------------------------------------------- criu
say "criu"
installed_major=0
if command -v criu >/dev/null; then
  installed=$(criu --version 2>/dev/null | head -1 | awk '{print $2}')
  installed_major=$(printf '%s' "$installed" | cut -d. -f1 | tr -dc '0-9')
  note "found criu ${installed:-unknown}"
fi

if [ "${installed_major:-0}" -ge "$CRIU_MIN_MAJOR" ] 2>/dev/null; then
  note "already at or above $CRIU_MIN_MAJOR.x, not rebuilding"
else
  note "need $CRIU_MIN_MAJOR.x or higher; distro packages ship 3.x without the CUDA plugin"
  say "installing build dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq --no-install-recommends \
    build-essential git pkg-config \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf libcap-dev libnl-3-dev libnet1-dev libaio-dev libbsd-dev \
    libnftables-dev libgnutls28-dev libdrm-dev uuid-dev >/dev/null 2>&1 \
    || die "dependency install failed"
  note "done"

  say "building criu $CRIU_VERSION"
  rm -rf "$BUILD_DIR"
  git clone -q --depth 1 --branch "$CRIU_VERSION" \
    https://github.com/checkpoint-restore/criu "$BUILD_DIR" \
    || die "clone failed"
  make -C "$BUILD_DIR" -j"$(nproc)" >/tmp/criu-build.log 2>&1 \
    || die "build failed; see /tmp/criu-build.log"
  note "built $("$BUILD_DIR"/criu/criu --version | head -1)"

  # install-man wants asciidoc and is not worth a dependency; install-cuda is
  # not wired at the top level, so the plugin is copied by hand below.
  make -C "$BUILD_DIR" install-criu install-lib install-compel \
    >/tmp/criu-install.log 2>&1 || die "install failed; see /tmp/criu-install.log"
  note "installed to $(command -v criu)"
fi

# ------------------------------------------------------------------ plugin
say "cuda plugin"
if [ -f "$PLUGIN_DIR/cuda_plugin.so" ]; then
  note "already at $PLUGIN_DIR/cuda_plugin.so"
elif [ -f "$BUILD_DIR/plugins/cuda/cuda_plugin.so" ]; then
  mkdir -p "$PLUGIN_DIR"
  install -m0755 "$BUILD_DIR/plugins/cuda/cuda_plugin.so" "$PLUGIN_DIR/"
  note "installed to $PLUGIN_DIR/cuda_plugin.so"
else
  die "no cuda_plugin.so: build criu from source first (remove $BUILD_DIR and re-run)"
fi

say "criu self-check"
if criu check >/dev/null 2>&1; then
  note "criu check: looks good"
else
  note "criu check: reported problems"
fi
if criu check --extra >/dev/null 2>&1; then
  note "criu check --extra: looks good"
else
  note "criu check --extra: some kernel features missing (dumps may still work)"
fi

# --------------------------------------------------------------- directories
say "runtime directories"
for d in /run/mncr /run/mncr/jobs /var/lib/mncr/images /var/lib/mncr/cache; do
  mkdir -p "$d" && note "$d"
done

# ----------------------------------------------------------------- verdict
if [ "$SKIP_PREFLIGHT" -eq 1 ]; then
  say "done (preflight skipped)"
  exit 0
fi

say "preflight"
here="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$here/agent/preflight.py" ]; then
  cd "$here" || die "cannot enter $here"
  MNCR_IMAGE_DIR=/var/lib/mncr/images MNCR_CACHE_DIR=/var/lib/mncr/cache \
  MNCR_JOBFILE_DIR=/run/mncr/jobs MNCR_CONTROL_ROOT=/run/mncr \
  PYTHONDONTWRITEBYTECODE=1 python3 -m agent.preflight --no-ipc
  exit $?
fi
note "agent/preflight.py not beside this script; skipping"
