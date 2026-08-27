"""Admission rules. Each one prevents a job that cannot be checkpointed from
starting in a place where nobody finds out until the first epoch."""

import unittest

from . import context  # noqa: F401
from k8s.admission import LABEL_CHECKPOINTABLE, NODE_SELECTOR_KEY, review

NODES = {
    "hgx-1": {"metadata": {"labels": {"mncr.io/driver-major": "610.57", "mncr.io/mnnvl": "false"}}},
    "nvl-9": {"metadata": {"labels": {"mncr.io/driver-major": "610.57", "mncr.io/mnnvl": "true"}}},
    "old-3": {"metadata": {"labels": {"mncr.io/driver-major": "580.1", "mncr.io/mnnvl": "false"}}},
}


def pod(node=None, checkpointable=True, selector=None):
    labels = {LABEL_CHECKPOINTABLE: "true"} if checkpointable else {}
    spec = {}
    if node:
        spec["nodeName"] = node
    if selector is not None:
        spec["nodeSelector"] = selector
    return {
        "request": {
            "uid": "uid-1",
            "object": {"metadata": {"labels": labels}, "spec": spec},
        }
    }


def decide(*args, **kwargs):
    return review(*args, node_lookup=NODES.get, **kwargs)["response"]


class TestAdmission(unittest.TestCase):
    def test_unlabelled_pods_are_ignored(self):
        response = decide(pod("nvl-9", checkpointable=False))
        self.assertTrue(response["allowed"])

    def test_checkpointable_pod_on_good_node_allowed(self):
        response = decide(pod("hgx-1", selector={NODE_SELECTOR_KEY: "true"}))
        self.assertTrue(response["allowed"])

    def test_mnnvl_node_denied(self):
        response = decide(pod("nvl-9", selector={NODE_SELECTOR_KEY: "true"}))
        self.assertFalse(response["allowed"])
        self.assertIn("fabric", response["status"]["message"])

    def test_mnnvl_node_allowed_when_explicitly_permitted(self):
        response = decide(pod("nvl-9", selector={NODE_SELECTOR_KEY: "true"}), allow_mnnvl=True)
        self.assertTrue(response["allowed"])

    def test_old_driver_denied(self):
        response = decide(pod("old-3", selector={NODE_SELECTOR_KEY: "true"}))
        self.assertFalse(response["allowed"])
        self.assertIn("below the minimum", response["status"]["message"])

    def test_unconstrained_pod_denied(self):
        response = decide(pod(node=None, selector={}))
        self.assertFalse(response["allowed"])
        self.assertIn("nodeSelector", response["status"]["message"])

    def test_selector_alone_is_enough_to_admit(self):
        response = decide(pod(node=None, selector={NODE_SELECTOR_KEY: "true"}))
        self.assertTrue(response["allowed"])

    def test_response_echoes_uid(self):
        self.assertEqual(decide(pod("hgx-1", selector={}))["uid"], "uid-1")


if __name__ == "__main__":
    unittest.main()
