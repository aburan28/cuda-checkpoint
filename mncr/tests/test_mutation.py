"""The mutating webhook.

Its whole value is removing a hand-written stanza people get wrong once. So the
tests are mostly about restraint: it must supply what is missing and never
overrule what an author already set.
"""

import base64
import json
import unittest

from . import context  # noqa: F401
from k8s.admission import CONTROL_PATH, CONTROL_VOLUME, mutate


def pod(containers=None, volumes=None, labels=None):
    metadata = {"labels": labels if labels is not None
                else {"mncr.io/checkpointable": "true", "job-name": "train-7"}}
    spec = {"containers": containers if containers is not None else [{"name": "trainer"}]}
    if volumes is not None:
        spec["volumes"] = volumes
    return {"request": {"uid": "u1", "object": {"metadata": metadata, "spec": spec}}}


def patches_of(response):
    body = response["response"]
    if "patch" not in body:
        return []
    return json.loads(base64.b64decode(body["patch"]))


def env_added(patches, container=0):
    out = {}
    for patch in patches:
        path = patch["path"]
        if path == f"/spec/containers/{container}/env":
            for entry in patch["value"]:
                out[entry["name"]] = entry["value"]
        elif path == f"/spec/containers/{container}/env/-":
            out[patch["value"]["name"]] = patch["value"]["value"]
    return out


class MutationTest(unittest.TestCase):
    def test_injects_volume_mount_and_env(self):
        patches = patches_of(mutate(pod()))
        paths = [p["path"] for p in patches]
        self.assertIn("/spec/volumes", paths)
        self.assertIn("/spec/containers/0/volumeMounts", paths)

        env = env_added(patches)
        self.assertEqual(env["MNCR_JOB_ID"], "train-7")
        self.assertEqual(env["MNCR_CONTROL_ROOT"], CONTROL_PATH)
        self.assertEqual(
            env["CUDA_CHECKPOINT_JOB_FILE"], f"{CONTROL_PATH}/jobs/train-7.jobfile"
        )
        self.assertTrue(env["MNCR_RANK_ADDR"].startswith("unix:"))
        # Measured on two nodes: without these three, a rank either cannot be
        # checkpointed at all or cannot be restored anywhere else.
        self.assertEqual(env["NCCL_RAS_ENABLE"], "0")
        self.assertEqual(env["FI_HMEM_CUDA_USE_GDRCOPY"], "0")
        self.assertEqual(env["LD_PRELOAD"], f"{CONTROL_PATH}/lib/libmncr_netmap.so")

    def test_ignores_pods_that_did_not_opt_in(self):
        response = mutate(pod(labels={}))
        self.assertNotIn("patch", response["response"])
        self.assertTrue(response["response"]["allowed"])

    def test_checkpointable_without_a_job_label_is_left_alone(self):
        response = mutate(pod(labels={"mncr.io/checkpointable": "true"}))
        self.assertNotIn("patch", response["response"])
        self.assertIn("no job label", response["response"]["status"]["message"])

    def test_explicit_job_id_label_wins_over_job_name(self):
        labels = {
            "mncr.io/checkpointable": "true",
            "mncr.io/job-id": "explicit",
            "job-name": "generated",
        }
        env = env_added(patches_of(mutate(pod(labels=labels))))
        self.assertEqual(env["MNCR_JOB_ID"], "explicit")

    def test_existing_env_is_never_overwritten(self):
        containers = [
            {
                "name": "trainer",
                "env": [{"name": "MNCR_JOB_ID", "value": "chosen-by-hand"}],
            }
        ]
        env = env_added(patches_of(mutate(pod(containers=containers))))
        self.assertNotIn("MNCR_JOB_ID", env, "the webhook overrode an author's value")
        self.assertIn("CUDA_CHECKPOINT_JOB_FILE", env)

    def test_existing_volume_is_not_duplicated(self):
        volumes = [{"name": CONTROL_VOLUME, "hostPath": {"path": CONTROL_PATH}}]
        paths = [p["path"] for p in patches_of(mutate(pod(volumes=volumes)))]
        self.assertNotIn("/spec/volumes", paths)
        self.assertNotIn("/spec/volumes/-", paths)

    def test_existing_mount_at_the_control_path_is_respected(self):
        containers = [
            {
                "name": "trainer",
                "volumeMounts": [{"name": "their-own", "mountPath": CONTROL_PATH}],
            }
        ]
        paths = [p["path"] for p in patches_of(mutate(pod(containers=containers)))]
        self.assertNotIn("/spec/containers/0/volumeMounts", paths)
        self.assertNotIn("/spec/containers/0/volumeMounts/-", paths)
        self.assertNotIn("/spec/volumes", paths)

    def test_every_container_is_configured(self):
        containers = [{"name": "trainer"}, {"name": "sidecar"}]
        patches = patches_of(mutate(pod(containers=containers)))
        self.assertTrue(env_added(patches, 0))
        self.assertTrue(env_added(patches, 1))

    def test_a_fully_configured_pod_gets_no_patch(self):
        containers = [
            {
                "name": "trainer",
                "volumeMounts": [{"name": CONTROL_VOLUME, "mountPath": CONTROL_PATH}],
                "env": [
                    {"name": "MNCR_JOB_ID", "value": "train-7"},
                    {"name": "MNCR_CONTROL_ROOT", "value": CONTROL_PATH},
                    {"name": "MNCR_RANK_ADDR", "value": "unix:/run/mncr/agent.sock"},
                    {"name": "CUDA_CHECKPOINT_JOB_FILE", "value": "/x"},
                    {"name": "NCCL_RAS_ENABLE", "value": "1"},
                    {"name": "FI_HMEM_CUDA_USE_GDRCOPY", "value": "1"},
                    {"name": "LD_PRELOAD", "value": "/my/own.so"},
                ],
            }
        ]
        volumes = [{"name": CONTROL_VOLUME, "hostPath": {"path": CONTROL_PATH}}]
        response = mutate(pod(containers=containers, volumes=volumes))
        self.assertNotIn("patch", response["response"])
        self.assertIn("already configured", response["response"]["status"]["message"])

    def test_patch_is_valid_jsonpatch(self):
        for patch in patches_of(mutate(pod())):
            self.assertEqual(patch["op"], "add")
            self.assertTrue(patch["path"].startswith("/spec/"))
            self.assertIn("value", patch)


if __name__ == "__main__":
    unittest.main()
