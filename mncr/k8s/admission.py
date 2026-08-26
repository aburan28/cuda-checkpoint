"""Validating admission for checkpointable workloads.

Two rules, both of which exist because the alternative is discovering the
problem after the commit point:

  1. A pod labelled checkpointable may not schedule onto a node exposing
     MNNVL/fabric state. Fabric handles cannot be checkpointed and there is no
     workaround, so a job that lands there is uncheckpointable and nobody finds
     out until the first epoch fails.

  2. A pod labelled checkpointable must tolerate only nodes at or above the
     minimum driver version, because restore across driver majors is not
     guaranteed.

Deployed as a ValidatingWebhookConfiguration. The handler is plain enough to
unit test without a cluster.
"""

import base64
import json

from mncr.version import MIN_DRIVER

LABEL_CHECKPOINTABLE = "mncr.io/checkpointable"
LABEL_DRIVER = "mncr.io/driver-major"
LABEL_MNNVL = "mncr.io/mnnvl"
NODE_SELECTOR_KEY = "mncr.io/checkpointable"


def _driver_major(value):
    try:
        return int(str(value).split(".")[0])
    except (ValueError, IndexError):
        return -1


def review(request, node_lookup=None, min_driver=MIN_DRIVER, allow_mnnvl=False):
    """Evaluate one AdmissionReview request. Returns the response dict."""
    uid = request.get("request", {}).get("uid", "")
    obj = request.get("request", {}).get("object", {}) or {}
    labels = (obj.get("metadata", {}) or {}).get("labels", {}) or {}

    if labels.get(LABEL_CHECKPOINTABLE, "false").lower() not in ("true", "1", "yes"):
        return _allow(uid, "not labelled checkpointable")

    spec = obj.get("spec", {}) or {}
    node_name = spec.get("nodeName")
    selector = spec.get("nodeSelector", {}) or {}

    # A checkpointable pod must be constrained to checkpointable nodes, whether
    # by explicit nodeName or by selector.
    if not node_name and NODE_SELECTOR_KEY not in selector:
        return _deny(
            uid,
            f"checkpointable pods must set nodeSelector {NODE_SELECTOR_KEY}=true so "
            f"they cannot land on a node that cannot checkpoint",
        )

    if node_name and node_lookup:
        node = node_lookup(node_name)
        node_labels = ((node or {}).get("metadata", {}) or {}).get("labels", {}) or {}

        # Knowing nothing about a node is not the same as knowing it is bad. If
        # the lookup came back empty - an API outage, a node that has not
        # registered yet - allow and say the check was not made. Denying here
        # would turn an API blip into a cluster-wide scheduling stop, and the
        # nodeSelector plus the agent's own preflight already keep a pod off a
        # node that cannot checkpoint.
        if not node_labels:
            return _allow(
                uid,
                f"node {node_name} could not be evaluated; allowing on the "
                f"nodeSelector and the agent preflight",
            )

        if not allow_mnnvl and node_labels.get(LABEL_MNNVL, "false").lower() == "true":
            return _deny(
                uid,
                f"node {node_name} exposes MNNVL/fabric state; fabric handles "
                f"cannot be checkpointed",
            )
        # A node labelled checkpointable but missing its driver label is a
        # labelling error, and that one is worth catching: it means something
        # produced a half-configured node.
        if LABEL_DRIVER not in node_labels:
            return _deny(
                uid,
                f"node {node_name} is labelled checkpointable but has no "
                f"{LABEL_DRIVER} label; it is only half configured",
            )
        major = _driver_major(node_labels[LABEL_DRIVER])
        if major < min_driver:
            return _deny(
                uid,
                f"node {node_name} driver major {major} is below the minimum "
                f"{min_driver} required for checkpoint/restore",
            )

    return _allow(uid, "checkpointable constraints satisfied")


def _allow(uid, message):
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": True,
            "status": {"message": message},
        },
    }


def _deny(uid, message):
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": False,
            "status": {"code": 403, "message": message},
        },
    }


def patch_response(uid, patches, message="mutated"):
    """An AdmissionReview response carrying a JSONPatch."""
    if not patches:
        return _allow(uid, message)
    encoded = base64.b64encode(json.dumps(patches).encode()).decode()
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": True,
            "patchType": "JSONPatch",
            "patch": encoded,
            "status": {"message": message},
        },
    }


# ------------------------------------------------------------------ mutation

CONTROL_VOLUME = "mncr-control"
CONTROL_PATH = "/run/mncr"
LABEL_JOB = "mncr.io/job-id"


def mutate(request, control_root=CONTROL_PATH, job_from_label=("mncr.io/job-id", "job-name")):
    """Give a checkpointable pod what it needs to be checkpointed.

    Three things, each of which is otherwise a hand-written stanza somebody
    forgets exactly once:

      the control mount   the rank polls it for its epoch request and the agent
                          writes there. hostPath, because agent and rank are
                          different pods
      MNCR_* env          job id, control root and the agent address, so
                          torchckpt.init() needs no arguments
      the job file path   driver 610 IPC, pointing at the file the agent creates
                          for this job

    Every operation is conditional. A pod that already sets something keeps what
    it set; the webhook is here to supply defaults, not to overrule authors.
    """
    uid = request.get("request", {}).get("uid", "")
    obj = request.get("request", {}).get("object", {}) or {}
    metadata = obj.get("metadata", {}) or {}
    labels = metadata.get("labels", {}) or {}

    if labels.get(LABEL_CHECKPOINTABLE, "false").lower() not in ("true", "1", "yes"):
        return _allow(uid, "not labelled checkpointable")

    job_id = next((labels[key] for key in job_from_label if labels.get(key)), None)
    if not job_id:
        return _allow(
            uid,
            f"checkpointable but no job label ({' or '.join(job_from_label)}); "
            f"nothing injected",
        )

    spec = obj.get("spec", {}) or {}
    patches = []
    notes = []

    volumes = spec.get("volumes")
    have_volume = any(v.get("name") == CONTROL_VOLUME for v in volumes or [])
    already_mounted = any(
        m.get("mountPath") == control_root
        for container in spec.get("containers", [])
        for m in container.get("volumeMounts", []) or []
    )

    if not have_volume and not already_mounted:
        volume = {
            "name": CONTROL_VOLUME,
            "hostPath": {"path": control_root, "type": "DirectoryOrCreate"},
        }
        if volumes is None:
            patches.append({"op": "add", "path": "/spec/volumes", "value": [volume]})
        else:
            patches.append({"op": "add", "path": "/spec/volumes/-", "value": volume})
        notes.append("control volume")

    for index, container in enumerate(spec.get("containers", [])):
        mounts = container.get("volumeMounts")
        mounted = any((m.get("mountPath") == control_root) for m in mounts or [])
        if not mounted and not already_mounted:
            mount = {"name": CONTROL_VOLUME, "mountPath": control_root}
            if mounts is None:
                patches.append(
                    {
                        "op": "add",
                        "path": f"/spec/containers/{index}/volumeMounts",
                        "value": [mount],
                    }
                )
            else:
                patches.append(
                    {
                        "op": "add",
                        "path": f"/spec/containers/{index}/volumeMounts/-",
                        "value": mount,
                    }
                )

        env = container.get("env")
        present = {e.get("name") for e in env or []}
        wanted = {
            "MNCR_JOB_ID": job_id,
            "MNCR_CONTROL_ROOT": control_root,
            "MNCR_RANK_ADDR": f"unix:{control_root}/agent.sock",
            "CUDA_CHECKPOINT_JOB_FILE": f"{control_root}/jobs/{job_id}.jobfile",
            # Two things measured to survive communicator teardown and fail a
            # restore: NCCL RAS keeps listeners per process, and libfabric
            # (aws-ofi-nccl) holds /dev/gdrdrv from plugin init on. Neither
            # can be released from inside the rank; both are prevented here.
            # An author who sets either keeps their value.
            "NCCL_RAS_ENABLE": "0",
            "FI_HMEM_CUDA_USE_GDRCOPY": "0",
            # NCCL keeps the node address it found at first init; a rank
            # restored on another node needs this shim to listen there. The
            # agent publishes it into the control directory on every node.
            "LD_PRELOAD": f"{control_root}/lib/libmncr_netmap.so",
        }
        missing = [
            {"name": name, "value": value}
            for name, value in wanted.items()
            if name not in present
        ]
        if not missing:
            continue
        if env is None:
            patches.append(
                {
                    "op": "add",
                    "path": f"/spec/containers/{index}/env",
                    "value": missing,
                }
            )
        else:
            for entry in missing:
                patches.append(
                    {
                        "op": "add",
                        "path": f"/spec/containers/{index}/env/-",
                        "value": entry,
                    }
                )
        notes.append(f"{len(missing)} env vars into {container.get('name', index)}")

    if not patches:
        return _allow(uid, "already configured for checkpointing")
    return patch_response(uid, patches, f"injected {'; '.join(notes)}")
