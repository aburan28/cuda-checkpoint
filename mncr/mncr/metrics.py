"""Prometheus metrics, stdlib only.

Small on purpose. The questions an operator actually asks of a checkpoint
system are few, and every one of them is answerable from a counter or a
histogram:

    how often do epochs abort, and how often are they lost past the commit
    point - these are different failures and must never share a series
    how long does the stop-the-world window last
    how much is being written, and how long does the driver take per rank
    are ranks voting dirty, and on what

Histogram buckets are seconds and chosen for this domain: a lock is
milliseconds, a checkpoint is seconds, a dump of a full node is minutes.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DURATION_BUCKETS = (0.05, 0.25, 1, 5, 15, 60, 300, 900, 3600)
BYTE_BUCKETS = (1 << 20, 1 << 24, 1 << 28, 1 << 30, 8 << 30, 64 << 30, 512 << 30)


def _key(labels):
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _render_labels(key, extra=None):
    pairs = list(key) + list(extra or ())
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    return "{" + inner + "}"


def _escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Metric:
    def __init__(self, name, help_text, kind):
        self.name = name
        self.help = help_text
        self.kind = kind
        self.lock = threading.Lock()


class Counter(_Metric):
    def __init__(self, name, help_text):
        super().__init__(name, help_text, "counter")
        self.values = {}

    def inc(self, amount=1, **labels):
        key = _key(labels)
        with self.lock:
            self.values[key] = self.values.get(key, 0) + amount

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self.lock:
            for key, value in sorted(self.values.items()):
                lines.append(f"{self.name}{_render_labels(key)} {value}")
        return lines


class Gauge(_Metric):
    def __init__(self, name, help_text):
        super().__init__(name, help_text, "gauge")
        self.values = {}

    def set(self, value, **labels):
        key = _key(labels)
        with self.lock:
            self.values[key] = value

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self.lock:
            for key, value in sorted(self.values.items()):
                lines.append(f"{self.name}{_render_labels(key)} {value}")
        return lines


class Histogram(_Metric):
    def __init__(self, name, help_text, buckets=DURATION_BUCKETS):
        super().__init__(name, help_text, "histogram")
        self.buckets = tuple(buckets)
        self.counts = {}
        self.sums = {}

    def observe(self, value, **labels):
        key = _key(labels)
        with self.lock:
            counts = self.counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[index] += 1
            counts[-1] += 1
            self.sums[key] = self.sums.get(key, 0.0) + value

    def time(self, **labels):
        return _Timer(self, labels)

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self.lock:
            for key, counts in sorted(self.counts.items()):
                # counts[i] is already cumulative: observe() increments every
                # bucket whose bound the value falls under.
                for index, bound in enumerate(self.buckets):
                    lines.append(
                        f"{self.name}_bucket"
                        f"{_render_labels(key, [('le', bound)])} {counts[index]}"
                    )
                lines.append(
                    f"{self.name}_bucket{_render_labels(key, [('le', '+Inf')])} "
                    f"{counts[-1]}"
                )
                lines.append(f"{self.name}_sum{_render_labels(key)} {self.sums[key]}")
                lines.append(f"{self.name}_count{_render_labels(key)} {counts[-1]}")
        return lines


class _Timer:
    def __init__(self, histogram, labels):
        self.histogram = histogram
        self.labels = labels
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.histogram.observe(time.perf_counter() - self.started, **self.labels)
        return False


class Registry:
    def __init__(self):
        self._metrics = []

    def add(self, metric):
        self._metrics.append(metric)
        return metric

    def counter(self, name, help_text):
        return self.add(Counter(name, help_text))

    def gauge(self, name, help_text):
        return self.add(Gauge(name, help_text))

    def histogram(self, name, help_text, buckets=DURATION_BUCKETS):
        return self.add(Histogram(name, help_text, buckets))

    def render(self):
        lines = []
        for metric in self._metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


REGISTRY = Registry()

# ------------------------------------------------------------------ series
EPOCHS = REGISTRY.counter(
    "mncr_epochs_total",
    "Checkpoint epochs by outcome. aborted and failed are different failures: "
    "aborted left the job intact, failed did not.",
)
EPOCH_SECONDS = REGISTRY.histogram(
    "mncr_epoch_seconds", "Wall clock of a whole epoch, by outcome."
)
PHASE_SECONDS = REGISTRY.histogram(
    "mncr_phase_seconds", "Time spent in each phase of an epoch."
)
STOPPED_SECONDS = REGISTRY.histogram(
    "mncr_stopped_seconds",
    "Stop-the-world window: from the first lock to the last unlock.",
)
VOTES = REGISTRY.counter("mncr_rank_votes_total", "Rank votes by verdict.")
DRIVER_SECONDS = REGISTRY.histogram(
    "mncr_driver_seconds", "Duration of one driver call, by action."
)
DRIVER_ERRORS = REGISTRY.counter(
    "mncr_driver_errors_total", "Failed driver calls, by action."
)
GATE_FINDINGS = REGISTRY.counter(
    "mncr_gate_findings_total",
    "Resources found at a verification gate that should not have been there.",
)
IMAGE_BYTES = REGISTRY.histogram(
    "mncr_image_bytes", "Stored image size per rank.", BYTE_BUCKETS
)
IMAGE_SECONDS = REGISTRY.histogram(
    "mncr_image_seconds", "Time to shard, compress and store one rank's image."
)
RANKS = REGISTRY.gauge("mncr_registered_ranks", "Ranks registered with this agent.")
POLICY_SUSPENDED = REGISTRY.gauge(
    "mncr_policy_suspended", "1 when a checkpoint policy has suspended itself."
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = REGISTRY.render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port=9180, host="0.0.0.0"):
    """Start the metrics endpoint on a background thread."""
    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, name="metrics", daemon=True)
    thread.start()
    return server
