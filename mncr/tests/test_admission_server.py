"""The webhook transport. The rules are tested in test_admission; these are the
properties of serving them - above all, that a broken webhook does not become a
cluster-wide scheduling outage."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from . import context  # noqa: F401
from k8s import admission_server


class FakeClient:
    namespace = "test"

    def __init__(self, nodes=None, fail=False):
        self._nodes = nodes or []
        self.fail = fail
        self.calls = 0

    def nodes(self):
        self.calls += 1
        if self.fail:
            from k8s.client import K8sError

            raise K8sError("api server unreachable")
        return self._nodes

    def get(self, path):
        from k8s.client import K8sError

        raise K8sError(f"no such object: {path}")


def node(name, mnnvl="false", driver="610.57"):
    return {
        "metadata": {
            "name": name,
            "labels": {"mncr.io/mnnvl": mnnvl, "mncr.io/driver-major": driver},
        }
    }


def pod_review(node_name, uid="u1"):
    return {
        "request": {
            "uid": uid,
            "object": {
                "metadata": {"labels": {"mncr.io/checkpointable": "true"}},
                "spec": {
                    "nodeName": node_name,
                    "nodeSelector": {"mncr.io/checkpointable": "true"},
                },
            },
        }
    }


class AdmissionServerTest(unittest.TestCase):
    def _serve(self, client):
        server = admission_server.build(port=0, client=client)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def _post(self, base, payload, path="/validate"):
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def test_allows_a_good_node(self):
        base = self._serve(FakeClient([node("hgx-1")]))
        response = self._post(base, pod_review("hgx-1"))["response"]
        self.assertTrue(response["allowed"])
        self.assertEqual(response["uid"], "u1")

    def test_denies_an_mnnvl_node(self):
        base = self._serve(FakeClient([node("nvl-9", mnnvl="true")]))
        response = self._post(base, pod_review("nvl-9"))["response"]
        self.assertFalse(response["allowed"])
        self.assertIn("fabric", response["status"]["message"])

    def test_health_endpoints(self):
        base = self._serve(FakeClient([]))
        for path in ("/healthz", "/readyz"):
            with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
                self.assertEqual(response.status, 200)

    def test_unknown_path_is_404(self):
        base = self._serve(FakeClient([]))
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(base, pod_review("hgx-1"), path="/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_malformed_body_is_400_not_a_crash(self):
        base = self._serve(FakeClient([]))
        request = urllib.request.Request(
            f"{base}/validate", data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(ctx.exception.code, 400)

    def test_api_server_outage_does_not_block_scheduling(self):
        """A webhook that cannot reach the API must not stop pods scheduling."""
        base = self._serve(FakeClient([], fail=True))
        response = self._post(base, pod_review("hgx-1"))["response"]
        self.assertTrue(response["allowed"])

    def test_unknown_node_is_allowed_not_denied(self):
        base = self._serve(FakeClient([]))       # node not in the list at all
        response = self._post(base, pod_review("ghost-node"))["response"]
        self.assertTrue(response["allowed"])
        self.assertIn("could not be evaluated", response["status"]["message"])

    def test_half_labelled_node_is_denied(self):
        """checkpointable but no driver label means something built it wrong."""
        partial = {"metadata": {"name": "half", "labels": {"mncr.io/mnnvl": "false"}}}
        base = self._serve(FakeClient([partial]))
        response = self._post(base, pod_review("half"))["response"]
        self.assertFalse(response["allowed"])
        self.assertIn("half configured", response["status"]["message"])

    def test_node_labels_are_cached_between_requests(self):
        client = FakeClient([node("hgx-1")])
        base = self._serve(client)
        for _ in range(5):
            self._post(base, pod_review("hgx-1"))
        self.assertEqual(client.calls, 1, "admission re-listed nodes on every request")


if __name__ == "__main__":
    unittest.main()
