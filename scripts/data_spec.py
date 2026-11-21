"""Reads the '## Loading' section of data/<dataset>/DATA_DESCRIPTION.md: how a dataset's files
   map onto modalities, sample ids, splits, and labels. Shared by loaders.py, profiler.py, and
   extract_embeddings.py, so there is one declaration per dataset and no per-dataset code.

   Format: one `key: value` per line. Values with several fields use `field=value | field=value`.

     modality.<name>: file=<glob> | kind=counts|image|text [| column=<column>] [| obs_key=<obs column>]
         One per modality. h5ad files: one file per modality, `column` unused.
         Parquet files: `file` may be a glob matching several shards, `column` names the column.
     id_column: <column>           parquet: the sample id column (h5ad: obs_names are the ids)
     split: filename_prefix        optional: split = file name up to the first '-' (e.g. train-0001.parquet)
     label.<name>: column=<column> [| first_code_of=<code>:<label>, <code>:<label>, ...]
         A label column. With first_code_of, the column holds a list of codes and the label is
         taken from the first listed code found in it (no match: missing label).

   Only lowercase keys with letters, digits, '_' and '.' are settings. Any other key in the
   section is an error, so a typo is never silently ignored.
"""
import glob
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
KINDS = {"counts", "image", "text"}
TOP_KEYS = {"id_column", "split"}
MODALITY_FIELDS = {"file", "kind", "column"}
LABEL_FIELDS = {"column", "first_code_of"}


def _fields(value, allowed, where):
    out = {}
    for part in value.split("|"):
        k, sep, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not sep or k not in allowed:
            raise ValueError(f"{where}: bad field '{part.strip()}' (allowed: {sorted(allowed)})")
        out[k] = v
    return out


def parse(dataset, description_root=DATA_ROOT):
    """Returns {"modalities": {name: {file, kind, column?}}, "labels": {name: {column, codes?}},
    "id_column": str|None, "split": str|None}."""
    path = Path(description_root) / dataset / "DATA_DESCRIPTION.md"
    if not path.is_file():
        raise ValueError(f"No {path}")
    text = path.read_text(encoding="utf-8")
    section = re.search(r"^##\s+Loading\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    if not section:
        raise ValueError(f"{path} has no '## Loading' section")

    spec = {"modalities": {}, "labels": {}, "id_column": None, "split": None}
    for line in section.group(1).splitlines():
        m = re.match(r"^\s*[-*]?\s*`?([a-z][a-z0-9_.]*)`?\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip("`")
        where = f"{path.name} '## Loading' line '{key}'"
        if key.startswith("modality."):
            f = _fields(value, MODALITY_FIELDS, where)
            if f.get("kind") not in KINDS or "file" not in f:
                raise ValueError(f"{where}: needs file= and kind= one of {sorted(KINDS)}")
            spec["modalities"][key.split(".", 1)[1]] = f
        elif key.startswith("label."):
            f = _fields(value, LABEL_FIELDS, where)
            if "column" not in f:
                raise ValueError(f"{where}: needs column=")
            if "first_code_of" in f:
                pairs = [p.strip() for p in f.pop("first_code_of").split(",") if p.strip()]
                f["codes"] = dict(p.split(":", 1) for p in pairs)
            spec["labels"][key.split(".", 1)[1]] = f
        elif key in TOP_KEYS:
            spec[key] = value
        else:
            raise ValueError(f"{where}: unknown setting (allowed: modality.<name>, label.<name>, {sorted(TOP_KEYS)})")
    if not spec["modalities"]:
        raise ValueError(f"{path}: '## Loading' declares no modality.<name> lines")
    return spec


def files_for(dataset, modality, spec, data_root=DATA_ROOT):
    """Sorted data files of one modality (a glob may match several shards)."""
    pattern = str(Path(data_root) / dataset / spec["modalities"][modality]["file"])
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise ValueError(f"{dataset}/{modality}: no file matches {pattern}")
    return files


def split_of(path, spec):
    if spec.get("split") == "filename_prefix":
        return Path(path).name.split("-", 1)[0]
    return None
