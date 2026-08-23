# syntax=docker/dockerfile:1
#
# Two targets: `agent` (privileged, per node) and `control` (coordinator,
# controller, admission). Both are stdlib-only Python, so the images carry no
# pip dependency tree - deliberate, because the agent runs privileged in
# somebody else's cluster and every dependency is one they have to audit.
#
#   docker build --target agent   -t mncr-agent:0.1.0 .
#   docker build --target control -t mncr-control:0.1.0 .
#
# The agent image needs three things from outside Python:
#   cuda-checkpoint   copied in from the vendor repo (a ~6 KB shim; it resolves
#                     everything through the driver at runtime)
#   criu >= 4.0       with cuda_plugin.so
#   libcuda           supplied by the host via the NVIDIA container runtime
#
# CRIU's CUDA plugin is the part that varies by distro. The distro package does
# not always include it; if `mncrctl preflight` reports it missing, build criu
# from source with BUILD_CRIU=1.

FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-minimal ca-certificates zstd \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/mncr
COPY mncr/ ./mncr/
COPY torchckpt/ ./torchckpt/
COPY coord/ ./coord/
COPY imagestore/ ./imagestore/
COPY k8s/ ./k8s/
COPY mncrctl ./mncrctl
ENV PYTHONPATH=/opt/mncr PYTHONUNBUFFERED=1 PATH=/opt/mncr:$PATH

# --------------------------------------------------------------- control
FROM base AS control
COPY verify/ ./verify/
LABEL org.opencontainers.image.title="mncr-control"
LABEL org.opencontainers.image.description="mncr coordinator, controller and admission webhook"
USER 65532:65532
ENTRYPOINT ["python3", "-m", "k8s.controller"]

# ----------------------------------------------------------------- agent
FROM base AS agent
ARG TARGETARCH=amd64
ARG BUILD_CRIU=0
RUN apt-get update \
 && apt-get install -y --no-install-recommends criu libcap2-bin \
 && rm -rf /var/lib/apt/lists/*

# Optional: criu from source, for distros whose package omits the CUDA plugin.
RUN if [ "$BUILD_CRIU" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler \
        protobuf-compiler python3-protobuf libcap-dev libnl-3-dev libnet1-dev \
        pkg-config libbsd-dev iproute2 libnftables-dev libgnutls28-dev \
      && git clone --depth 1 --branch v4.0 https://github.com/checkpoint-restore/criu /tmp/criu \
      && make -C /tmp/criu -j"$(nproc)" \
      && make -C /tmp/criu install \
      && make -C /tmp/criu/plugins/cuda install \
      && rm -rf /tmp/criu /var/lib/apt/lists/*; \
    fi

COPY agent/ ./agent/
COPY verify/ ./verify/
COPY audit/ ./audit/
# The vendor utility, from the repo this lives in.
COPY --chmod=0755 vendor-bin/cuda-checkpoint /usr/local/bin/cuda-checkpoint

LABEL org.opencontainers.image.title="mncr-agent"
LABEL org.opencontainers.image.description="mncr per-node agent: driver calls, CRIU, job files"
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python3 -c "import socket,sys; s=socket.create_connection(('127.0.0.1',7181),3); s.close()" || exit 1
ENTRYPOINT ["python3", "-m", "agent.main"]
