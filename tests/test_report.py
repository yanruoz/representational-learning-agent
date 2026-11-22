"""Report rules that don't need a full run: the page cap, readable labels, merged identical baselines.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import labels  # noqa: E402
import report  # noqa: E402


class TestReport(unittest.TestCase):
    def test_page_cap(self):
        report.check_page_cap(report.MAX_PAGES)
        with self.assertRaises(SystemExit):
            report.check_page_cap(report.MAX_PAGES + 1)

    def test_labels_are_readable_and_unique(self):
        plan = {"models": [
            {"id": "a", "method": "two_tower", "loss": "infonce", "inputs": {"img": {"encoder": "cache:dinov2_vitb14_frozen"}, "txt": {"encoder": "tfidf_svd"}}},
            {"id": "b", "method": "two_tower", "loss": "infonce", "inputs": {"img": {"encoder": "cache:dinov2_vitb14_frozen"}, "txt": {"encoder": "cache:biomedclip_frozen"}}},
            {"id": "c", "method": "frozen_probe", "inputs": {"img": {"encoder": "cache:dinov2_vitb14_frozen"}}},
            {"id": "d", "method": "frozen_probe", "inputs": {"img": {"encoder": "cache:biomedclip_frozen"}}}]}
        lab = labels.model_labels(plan, {"inference_modalities": ["img"]})
        self.assertEqual(len(set(lab.values())), 4)
        self.assertIn("InfoNCE", lab["a"])
        self.assertIn("control", lab["c"])
        self.assertIn("reference", lab["d"])

    def test_identical_baselines_merge(self):
        m = {"x.metric": {"mean": 0.5, "sd": 0.0, "n": 1}}
        summary = {"p": {"model_id": "p", "method": "single_modality_features", "metrics": m},
                   "q": {"model_id": "q", "method": "linear_supervised", "metrics": dict(m)},
                   "r": {"model_id": "r", "method": "two_tower", "metrics": dict(m)}}
        rows = labels.merge_identical(summary, {"p": "P", "q": "Q", "r": "R"})
        self.assertEqual(len(rows), 2)  # baselines merged, the learned model stays separate

    def test_sig3(self):
        self.assertEqual(labels.sig3(0.010587), "0.0106")
        self.assertEqual(labels.sig3(0.59943), "0.599")


if __name__ == "__main__":
    unittest.main()
