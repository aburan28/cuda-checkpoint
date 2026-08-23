"""HTTPS server for the validating webhook.

`admission.py` holds the decision and is unit tested without a cluster; this is
only the transport. Kept separate on purpose - the rules are the part worth
reading, and they should not be buried in socket handling.

Serving TLS is not optional: the API server will not talk to a webhook over
plain HTTP. Certificates come from a mounted secret.
"""

import argparse
import json
import os
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mncr import log  # noqa: E402
from mncr.version import MIN_DRIVER  # noqa: E402

from .admission import review  # noqa: E402
from .client import K8sClient, K8sError  # noqa: E402

_LOG = log.get("k8s.admission")


class NodeCache:
    """Nodes change slowly and admission is on the scheduling path.

    A stale label is a decision made on old information, so the TTL is short
    and a miss always falls through to a live read.
    """

    def __init__(self, client, ttl=30.0):
        self.client = client
        self.ttl = ttl
        self._lock = threading.Lock()
        self._nodes = {}
        self._fetched = 0.0

    def _refresh(self):
        import time

        with self._lock:
            if time.time() - self._fetched < self.ttl and self._nodes:
                return
            try:
                nodes = self.client.nodes()
            except K8sError as exc:
                _LOG.warn("node list failed; serving stale labels", error=str(exc))
                return
            self._nodes = {n["metadata"]["name"]: n for n in nodes}
            self._fetched = time.time()

    def get(self, name):
        self._refresh()
        with self._lock:
            node = self._nodes.get(name)
        if node is not None:
            return node
        try:
            return self.client.get(f"/api/v1/nodes/{name}")
        except K8sError:
            return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass    # the structured logger handles this

    def _respond(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/healthz", "/readyz"):
            self._respond(200, {"ok": True})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != self.server.path:
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._respond(400, {"error": f"malformed AdmissionReview: {exc}"})
            return

        try:
            response = review(
                request,
                node_lookup=self.server.nodes.get,
                min_driver=self.server.min_driver,
                allow_mnnvl=self.server.allow_mnnvl,
            )
        except Exception as exc:  # noqa: BLE001
            # An admission controller that throws blocks scheduling. Allow, and
            # be loud about why the decision was not made.
            _LOG.error("review raised; allowing", error=str(exc))
            uid = request.get("request", {}).get("uid", "")
            response = {
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": True,
                    "status": {"message": f"admission error, allowed: {exc}"},
                },
            }

        decision = response["response"]
        _LOG.info(
            "review",
            allowed=decision["allowed"],
            message=decision.get("status", {}).get("message", "")[:120],
        )
        self._respond(200, response)


def build(port=8443, path="/validate", certfile=None, keyfile=None, client=None,
          allow_mnnvl=False, min_driver=MIN_DRIVER):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.path = path
    server.nodes = NodeCache(client or K8sClient())
    server.allow_mnnvl = allow_mnnvl
    server.min_driver = min_driver

    if certfile and keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    else:
        _LOG.warn(
            "serving without TLS; the API server will refuse to call this. "
            "Development only."
        )
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(description="mncr admission webhook")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--path", default="/validate")
    ap.add_argument("--cert", default=os.environ.get("MNCR_TLS_CERT", "/tls/tls.crt"))
    ap.add_argument("--key", default=os.environ.get("MNCR_TLS_KEY", "/tls/tls.key"))
    ap.add_argument("--allow-mnnvl", action="store_true")
    args = ap.parse_args(argv)

    cert = args.cert if os.path.exists(args.cert) else None
    key = args.key if os.path.exists(args.key) else None
    server = build(args.port, args.path, cert, key, allow_mnnvl=args.allow_mnnvl)
    _LOG.info("admission listening", port=args.port, path=args.path, tls=bool(cert))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
