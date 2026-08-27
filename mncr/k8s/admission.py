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
        node = node_lookup(node_name) or {}
        node_labels = (node.get("metadata", {}) or {}).get("labels", {}) or {}
        if not allow_mnnvl and node_labels.get(LABEL_MNNVL, "false").lower() == "true":
            return _deny(
                uid,
                f"node {node_name} exposes MNNVL/fabric state; fabric handles "
                f"cannot be checkpointed",
            )
        major = _driver_major(node_labels.get(LABEL_DRIVER, "0"))
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
    """Helper for a mutating variant that injects the job file env var."""
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
