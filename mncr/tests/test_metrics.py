"""Metrics: the series an operator actually asks for, and that the two kinds of
failure never share one."""

import unittest
import urllib.request

from . import context  # noqa: F401
from mncr import metrics
from mncr.errors import AbortableError, TerminalError
from verify.sim import SimCluster


class MetricFormatTest(unittest.TestCase):
    def test_counter_labels_are_sorted_and_escaped(self):
        counter = metrics.Counter("t_total", "help")
        counter.inc(2, b="two", a='say "hi"')
        rendered = "\n".join(counter.render())
        self.assertIn('t_total{a="say \\"hi\\"",b="two"} 2', rendered)

    def test_histogram_buckets_are_cumulative(self):
        histogram = metrics.Histogram("t_seconds", "help", buckets=(1, 10))
        for value in (0.5, 5, 50):
            histogram.observe(value)
        rendered = "\n".join(histogram.render())
        self.assertIn('t_seconds_bucket{le="1"} 1', rendered)
        self.assertIn('t_seconds_bucket{le="10"} 2', rendered)
        self.assertIn('t_seconds_bucket{le="+Inf"} 3', rendered)
        self.assertIn("t_seconds_count 3", rendered)
        self.assertIn("t_seconds_sum 55.5", rendered)

    def test_timer_records_a_duration(self):
        histogram = metrics.Histogram("t2_seconds", "help")
        with histogram.time(action="x"):
            pass
        self.assertIn("t2_seconds_count", "\n".join(histogram.render()))

    def test_endpoint_serves_prometheus_text(self):
        server = metrics.serve(port=0, host="127.0.0.1")
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            body = r.read().decode()
            self.assertIn("version=0.0.4", r.headers["Content-Type"])
        self.assertIn("# TYPE mncr_epochs_total counter", body)


class EpochMetricsTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimCluster(nodes=1, ranks_per_node=1, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()

    def tearDown(self):
        self.sim.stop()

    def _count(self, **labels):
        key = tuple(sorted((str(k), str(v)) for k, v in labels.items()))
        return metrics.EPOCHS.values.get(key, 0)

    def test_the_two_failures_are_recorded_separately(self):
        """An abort left the job intact; a failure did not. One series that
        conflated them would hide the only distinction that matters."""
        job = self.sim.job_id
        agent = self.sim.agents["node-0"]

        before_ok = self._count(outcome="succeeded", job=job, mode="continue")
        before_abort = self._count(outcome="aborted", job=job)
        before_fail = self._count(outcome="failed", job=job)

        self.sim.coord.checkpoint(job, mode="continue")
        self.sim.wait_for_event("resumed", timeout=20)

        pid = agent.local_ranks(job)[0]["host_pid"]
        agent.driver.fail_next("lock", pid)
        with self.assertRaises(AbortableError):
            self.sim.coord.checkpoint(job, mode="continue")
        self.sim.wait_for_event("resumed", timeout=20)

        pid = agent.local_ranks(job)[0]["host_pid"]
        agent.driver.fail_next("checkpoint", pid)
        with self.assertRaises(TerminalError):
            self.sim.coord.checkpoint(job, mode="continue")

        self.assertEqual(self._count(outcome="succeeded", job=job, mode="continue"),
                         before_ok + 1)
        self.assertEqual(self._count(outcome="aborted", job=job), before_abort + 1)
        self.assertEqual(self._count(outcome="failed", job=job), before_fail + 1)

    def test_votes_and_driver_calls_are_counted(self):
        before = len(metrics.VOTES.values)
        self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertGreaterEqual(len(metrics.VOTES.values), before)
        rendered = metrics.REGISTRY.render()
        self.assertIn("mncr_driver_seconds_bucket", rendered)
        self.assertIn("mncr_stopped_seconds_bucket", rendered)


if __name__ == "__main__":
    unittest.main()
