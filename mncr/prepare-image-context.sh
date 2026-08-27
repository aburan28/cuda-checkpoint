#!/usr/bin/env bash
# Stage the vendor cuda-checkpoint binary into the build context.
#
# The Dockerfile cannot COPY from outside its context, and the binary lives in
# the parent repo. Run this before `docker build`.
set -euo pipefail

arch="${1:-$(uname -m)}"
case "$arch" in
  x86_64|amd64)   src="../bin/x86_64_Linux/cuda-checkpoint" ;;
  aarch64|arm64)  src="../bin/aarch64_Linux/cuda-checkpoint" ;;
  *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac

if [ ! -f "$src" ]; then
  echo "not found: $src" >&2
  echo "Run this from the mncr/ directory inside the cuda-checkpoint repo." >&2
  exit 1
fi

mkdir -p vendor-bin
cp "$src" vendor-bin/cuda-checkpoint
chmod +x vendor-bin/cuda-checkpoint
echo "staged $src -> vendor-bin/cuda-checkpoint"
