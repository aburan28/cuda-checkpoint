"""Device map construction. A partial map is a rejected restore."""

import unittest

from . import context  # noqa: F401
from coord.devicemap import (
    build_pairs,
    describe,
    identity,
    is_identity,
    normalize,
    to_api,
    to_cli,
)
from mncr.errors import PlacementError


class TestDeviceMap(unittest.TestCase):
    def test_normalize_adds_prefix_once(self):
        self.assertEqual(normalize("abc"), "GPU-abc")
        self.assertEqual(normalize("GPU-abc"), "GPU-abc")

    def test_positional_pairing(self):
        pairs = build_pairs(["a", "b", "c"], ["x", "y", "z"])
        self.assertEqual(pairs[1], ("GPU-b", "GPU-y"))

    def test_count_mismatch_rejected(self):
        with self.assertRaises(PlacementError) as ctx:
            build_pairs(["a", "b"], ["x"])
        self.assertIn("every visible GPU must be mapped", str(ctx.exception))

    def test_empty_source_rejected(self):
        with self.assertRaises(PlacementError):
            build_pairs([], ["x"])

    def test_cli_format(self):
        self.assertEqual(to_cli(build_pairs(["a"], ["b"])), "GPU-a=GPU-b")

    def test_api_format_matches_driver_struct(self):
        self.assertEqual(
            to_api(build_pairs(["a"], ["b"])),
            [{"oldUuid": "GPU-a", "newUuid": "GPU-b"}],
        )

    def test_identity_map_is_explicit_not_empty(self):
        pairs = identity(["a", "b"])
        self.assertTrue(is_identity(pairs))
        self.assertEqual(to_cli(pairs), "GPU-a=GPU-a,GPU-b=GPU-b")

    def test_describe_counts_moves(self):
        info = describe(build_pairs(["a", "b"], ["a", "z"]))
        self.assertEqual((info["devices"], info["moved"], info["identity"]), (2, 1, False))


if __name__ == "__main__":
    unittest.main()
