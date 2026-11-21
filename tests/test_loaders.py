"""Loader, spec, and profiler tests on small synthetic datasets (no real data needed).

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import data_spec  # noqa: E402
import loaders  # noqa: E402
import profiler  # noqa: E402

FILES = "## Files\n- file: x | url: https://example.org/x | md5: 0\n"


def write_desc(root, name, loading):
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "DATA_DESCRIPTION.md").write_text(f"# {name}\n\n## Loading\n{loading}\n{FILES}")


class TestSpec(unittest.TestCase):
    def test_typo_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            write_desc(Path(d), "t", "- modalty.a: file=a.h5ad | kind=counts\n")
            with self.assertRaises(ValueError):
                data_spec.parse("t", d)

    def test_codes_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            write_desc(Path(d), "t", "- modality.a: file=*.parquet | kind=text | column=t\n"
                                     "- label.m: column=c | first_code_of=X1:alpha, X2:beta\n")
            spec = data_spec.parse("t", d)
            self.assertEqual(spec["labels"]["m"]["codes"], {"X1": "alpha", "X2": "beta"})


class TestParquet(unittest.TestCase):
    def test_paired_load_labels_and_profile(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        def png(v):
            b = io.BytesIO()
            Image.new("L", (8, 6), color=v).save(b, format="PNG")
            return {"bytes": b.getvalue(), "path": None}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_desc(root, "t", "- modality.img: file=*.parquet | kind=image | column=image\n"
                                  "- modality.txt: file=*.parquet | kind=text | column=caption\n"
                                  "- id_column: id\n- split: filename_prefix\n"
                                  "- label.m: column=codes | first_code_of=X1:alpha, X2:beta\n")
            for split, ids in (("train", ["a", "b", "c"]), ("test", ["d"])):
                tab = pa.table({"id": ids, "image": [png(i * 40) for i in range(len(ids))],
                                "caption": [f"an alpha view {i}" for i in ids],
                                "codes": [["X2", "Z"], ["Z", "X1"], ["nan"], ["X1"]][: len(ids)]})
                pq.write_table(tab, root / "t" / f"{split}-0.parquet")
            out = loaders.load("t", d, d)
            self.assertEqual(sorted(out["ids"]), ["a", "b", "c", "d"])
            self.assertEqual(dict(zip(out["ids"], out["split"]))["d"], "test")
            labels = dict(zip(out["ids"], out["labels"]["m"]))
            self.assertEqual(labels["a"], "beta")    # first listed code found
            self.assertEqual(labels["b"], "alpha")   # found in a later position
            self.assertIsNone(labels["c"])           # no listed code: missing
            imgs = [im for _, batch in loaders.iter_images(out["modalities"]["img"]["refs"], "image", 2) for im in batch]
            self.assertEqual([im.size for im in imgs], [(8, 6)] * 4)

            p = profiler.profile("t", root, root)
            self.assertEqual(p["labels"]["m"]["n_missing"], 1)
            self.assertEqual(p["modalities"]["img"]["modes"], {"L": 4})


class TestH5ad(unittest.TestCase):
    def test_mismatched_ids_rejected(self):
        import anndata as ad
        import scipy.sparse as sp

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_desc(root, "t", "- modality.a: file=a.h5ad | kind=counts\n- modality.b: file=b.h5ad | kind=counts\n")
            X = sp.csr_matrix(np.eye(3, dtype=np.float32))
            ad.AnnData(X, obs={"_": [0, 1, 2]}).write_h5ad(root / "t" / "a.h5ad")
            b = ad.AnnData(X, obs={"_": [0, 1, 2]})
            b.obs_names = ["x", "y", "z"]
            b.write_h5ad(root / "t" / "b.h5ad")
            with self.assertRaises(ValueError):
                loaders.load("t", d, d)


if __name__ == "__main__":
    unittest.main()
