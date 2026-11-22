"""Evidence resolution used by every validator: dotted keys whose parts may contain dots or symbols,
and values matched at the precision quoted.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evidence  # noqa: E402


class TestEvidence(unittest.TestCase):
    def test_resolve_keys_with_dots_and_symbols(self):
        obj = {"m": {"metrics": {"cross_modal_retrieval.a->b.recall@10": {"mean": 0.4708, "sd": 0.003}},
                     "x": {"fraction_above_0.3": 0.25}}}
        ok, v = evidence.resolve(obj, "m.metrics.cross_modal_retrieval.a->b.recall@10")
        self.assertTrue(ok)
        self.assertEqual(v["mean"], 0.4708)
        ok, v = evidence.resolve(obj, "m.x.fraction_above_0.3")
        self.assertTrue(ok and v == 0.25)
        self.assertFalse(evidence.resolve(obj, "m.metrics.missing")[0])

    def test_matches_at_quoted_precision(self):
        self.assertTrue(evidence.matches(0.47, {"mean": 0.4708}))
        self.assertTrue(evidence.matches(0.4708, 0.47081))
        self.assertFalse(evidence.matches(0.48, 0.4708))
        self.assertTrue(evidence.matches("lsi", "lsi"))

    def test_banned_reasons(self):
        errors = []
        evidence.check_reason("the default choice for this kind of data", "r", errors)
        self.assertTrue(errors)
        errors = []
        evidence.check_reason("short", "r", errors)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
