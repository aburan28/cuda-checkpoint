"""Minimal Kubernetes API client.

urllib against the in-cluster endpoint with the service account token. The agent
runs privileged in somebody else's cluster; every dependency it carries is one
they have to audit, and the surface used here is six verbs.

Falls back to `kubectl` when not running in a cluster, so the controller can be
driven against a kubeconfig during development.
"""

import json
import os
import ssl
import subprocess
import urllib.error
import urllib.request

from mncr import log

_LOG = log.get("k8s.client")

SA_ROOT = "/var/run/secrets/kubernetes.io/serviceaccount"


class K8sError(Exception):
    pass


class K8sClient:
    def __init__(self, base=None, token=None, ca_path=None, namespace=None):
        self.in_cluster = os.path.exists(os.path.join(SA_ROOT, "token"))
        if self.in_cluster:
            self.base = base or (
                f"https://{os.environ.get('KUBERNETES_SERVICE_HOST', 'kubernetes.default')}"
                f":{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}"
            )
            self.token = token or open(os.path.join(SA_ROOT, "token")).read().strip()
            self.ca_path = ca_path or os.path.join(SA_ROOT, "ca.crt")
            self.namespace = namespace or open(
                os.path.join(SA_ROOT, "namespace")
            ).read().strip()
            self._ctx = ssl.create_default_context(cafile=self.ca_path)
        else:
            self.base = base
            self.token = token
            self.namespace = namespace or "default"
            self._ctx = None

    # ------------------------------------------------------------- transport
    def _request(self, method, path, body=None, stream=False, timeout=60):
        if not self.in_cluster:
            return self._kubectl(method, path, body)
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            resp = urllib.request.urlopen(req, context=self._ctx, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise K8sError(f"{method} {path} -> {exc.code}: {exc.read()[:300]}") from exc
        if stream:
            return resp
        return json.loads(resp.read() or b"{}")

    def _kubectl(self, method, path, body):
        """Development path. Not used in cluster."""
        if method == "GET":
            proc = subprocess.run(
                ["kubectl", "get", "--raw", path], capture_output=True, text=True
            )
        elif method in ("POST", "PUT", "PATCH"):
            proc = subprocess.run(
                ["kubectl", "create" if method == "POST" else "replace", "-f", "-"],
                input=json.dumps(body),
                capture_output=True,
                text=True,
            )
        else:
            proc = subprocess.run(
                ["kubectl", "delete", "--raw", path], capture_output=True, text=True
            )
        if proc.returncode != 0:
            raise K8sError(f"kubectl {method} {path}: {proc.stderr.strip()[:300]}")
        try:
            return json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------ verbs
    def list(self, path):
        return self._request("GET", path).get("items", [])

    def get(self, path):
        return self._request("GET", path)

    def patch_status(self, path, status):
        """Merge-patch a custom resource status subresource."""
        url = f"{path}/status"
        if not self.in_cluster:
            return self._kubectl("PATCH", url, {"status": status})
        data = json.dumps({"status": status}).encode()
        req = urllib.request.Request(f"{self.base}{url}", data=data, method="PATCH")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/merge-patch+json")
        try:
            resp = urllib.request.urlopen(req, context=self._ctx, timeout=30)
        except urllib.error.HTTPError as exc:
            raise K8sError(f"PATCH {url} -> {exc.code}: {exc.read()[:300]}") from exc
        return json.loads(resp.read() or b"{}")

    def watch(self, path, resource_version=None, timeout=300):
        """Yield watch events. Reconnects are the caller's business."""
        sep = "&" if "?" in path else "?"
        url = f"{path}{sep}watch=true&timeoutSeconds={timeout}"
        if resource_version:
            url += f"&resourceVersion={resource_version}"
        resp = self._request("GET", url, stream=True, timeout=timeout + 30)
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                _LOG.warn("undecodable watch event")

    # ------------------------------------------------------------- shortcuts
    def nodes(self, selector=None):
        path = "/api/v1/nodes"
        if selector:
            path += f"?labelSelector={selector}"
        return self.list(path)

    def pods(self, namespace=None, selector=None):
        ns = namespace or self.namespace
        path = f"/api/v1/namespaces/{ns}/pods"
        if selector:
            path += f"?labelSelector={selector}"
        return self.list(path)

    def delete_pod(self, name, namespace=None, grace=0):
        ns = namespace or self.namespace
        return self._request(
            "DELETE", f"/api/v1/namespaces/{ns}/pods/{name}?gracePeriodSeconds={grace}"
        )

    def crs(self, plural, namespace=None, group="mncr.io", version="v1alpha1"):
        ns = namespace or self.namespace
        return self.list(f"/apis/{group}/{version}/namespaces/{ns}/{plural}")

    def cr_path(self, plural, name, namespace=None, group="mncr.io", version="v1alpha1"):
        ns = namespace or self.namespace
        return f"/apis/{group}/{version}/namespaces/{ns}/{plural}/{name}"
