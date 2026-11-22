"""Evidence items: {"file": "<path under outputs/<d>/>", "key": "a.b.c", "value": <number or string>}.
   Shared by the validators: every reason that cites evidence must quote a value that matches the file.

   Keys are dotted paths. Some keys contain dots or other symbols themselves ("recall@10",
   "cache:dinov2_vitb14_frozen", "fraction_features_r_above_0.3"), so resolution tries the longest
   key that matches at each level.
"""
import json

import data_spec

ALLOWED_FILES = {"profile.json", "metrics/baseline_check.json", "metrics/summary.json", "metrics/collapse.json", "run_log.json"}
BANNED_REASONS = ("default", "standard choice", "commonly used", "common choice", "as usual", "typical choice")


def resolve(obj, key):
    parts = key.split(".")

    def walk(o, i):
        if i == len(parts):
            return True, o
        if not isinstance(o, dict):
            return False, None
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if k in o:
                ok, v = walk(o[k], j)
                if ok:
                    return True, v
        return False, None

    return walk(obj, 0)


def matches(quoted, actual):
    """Numbers match at the precision quoted (0.81 matches 0.8137); strings must be equal."""
    if isinstance(actual, dict) and "mean" in actual:
        actual = actual["mean"]
    if isinstance(quoted, (int, float)) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        s = repr(float(quoted))
        dec = len(s.split(".")[1]) if "." in s and not s.endswith(".0") else 0
        return round(float(actual), dec) == round(float(quoted), dec)
    return quoted == actual


def check_items(items, dataset, where, errors):
    """Validate a list of evidence items; append messages to errors. Returns the number checked."""
    if not isinstance(items, list) or not items:
        errors.append(f"{where}: needs a non-empty 'evidence' list ({{file, key, value}} items)")
        return 0
    out = data_spec.REPO / "outputs" / dataset
    for k, it in enumerate(items):
        f, key = it.get("file"), it.get("key")
        if f not in ALLOWED_FILES:
            errors.append(f"{where}: evidence[{k}].file must be one of {sorted(ALLOWED_FILES)}, got {f!r}")
            continue
        path = out / f
        if not path.exists():
            errors.append(f"{where}: evidence[{k}] cites {f}, which does not exist yet")
            continue
        ok, actual = resolve(json.loads(path.read_text()), key or "")
        if not ok:
            errors.append(f"{where}: evidence[{k}] key {key!r} not found in {f}")
        elif "value" not in it or not matches(it["value"], actual):
            errors.append(f"{where}: evidence[{k}] quotes {it.get('value')!r} for {f}:{key}, but the file has {actual!r}")
    return len(items)


def check_reason(text, where, errors, min_len=25):
    if not isinstance(text, str) or len(text.strip()) < min_len:
        errors.append(f"{where}: needs a specific reason (at least {min_len} characters)")
        return
    low = text.lower()
    for b in BANNED_REASONS:
        if b in low:
            errors.append(f"{where}: '{b}' is never a reason; say why it fits this data and objective")
