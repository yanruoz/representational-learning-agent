"""Setup step, deliberately separate from an analysis run: downloads every data file a
   dataset declares in the '## Files' section of its DATA_DESCRIPTION.md, if missing,
   and verifies each file's checksum. No dataset is special-cased.

   '## Files' format, one file per line (md5 or sha256, at least one is required):
       - file: <name> | url: <http(s) url> | sha256: <hex>
       - file: <name> | url: <http(s) url> | md5: <hex>

   The analysis pipeline never touches the network, so a fresh clone runs this once:
       venv/bin/python scripts/fetch_data.py --dataset <name>
   On a cluster, put the files somewhere else (the description is still read from the repo):
       python scripts/fetch_data.py --dataset <name> --data-root /work/.../data

   Standard library only, so it runs before any environment is set up.
"""
import argparse
import hashlib
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESCRIPTIONS = REPO / "data"
ALLOWED_SCHEMES = ("http://", "https://")
FIELDS = {"file", "url", "md5", "sha256"}

# Some hosts return HTTP 403 for urllib's default "Python-urllib/x.y" User-Agent.
USER_AGENT = "rep-agent-fetch/1.0"


def parse_files_section(description_path):
    """Returns a list of {file, url, md5?/sha256?} dicts from the '## Files' section."""
    text = Path(description_path).read_text(encoding="utf-8")
    section = re.search(r"^##\s+Files\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    if not section:
        raise ValueError(f"{description_path} has no '## Files' section")
    entries = []
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line.startswith(("-", "*")) or "file:" not in line:
            continue
        entry = {}
        for part in line.lstrip("-* ").split("|"):
            key, sep, value = part.partition(":")
            key, value = key.strip().strip("`"), value.strip().strip("`")
            if not sep:
                continue
            if key not in FIELDS:
                raise ValueError(f"Unknown field '{key}' in '{line}' (allowed: {sorted(FIELDS)})")
            entry[key] = value
        if "file" not in entry or "url" not in entry:
            raise ValueError(f"Line needs both 'file:' and 'url:': {line}")
        if not entry["url"].startswith(ALLOWED_SCHEMES):
            raise ValueError(f"url must start with one of {ALLOWED_SCHEMES}: {entry['url']}")
        if "md5" not in entry and "sha256" not in entry:
            raise ValueError(f"Line needs an 'md5:' or 'sha256:' checksum: {line}")
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"'file:' must be a plain file name, not a path: {entry['file']}")
        entries.append(entry)
    if not entries:
        raise ValueError(f"'## Files' in {description_path} lists no files")
    return entries


def checksum(path, algo, chunk=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, entry):
    """Returns None if every declared checksum matches, else a message."""
    for algo in ("sha256", "md5"):
        if algo in entry:
            actual = checksum(path, algo)
            if actual != entry[algo].lower():
                return f"{algo} {actual} does not match the declared {entry[algo]}"
    return None


def fetch_one(entry, folder):
    target = folder / entry["file"]
    if target.exists():
        problem = verify(target, entry)
        if problem:
            raise ValueError(f"{target} exists but its {problem}. Not overwriting; delete it yourself to re-download.")
        return f"already present (verified): {target}"

    partial = target.with_name(target.name + ".part")
    print(f"downloading {entry['url']} -> {target}", flush=True)
    try:
        req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp, open(partial, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
    except Exception as e:
        partial.unlink(missing_ok=True)
        raise ValueError(f"download of {entry['url']} failed: {e}") from e

    problem = verify(partial, entry)
    if problem:
        partial.unlink(missing_ok=True)
        raise ValueError(f"downloaded {entry['file']}: {problem}; discarded it.")
    partial.rename(target)
    return f"downloaded (verified): {target}"


def fetch(dataset, data_root=None):
    description = DESCRIPTIONS / dataset / "DATA_DESCRIPTION.md"
    if not description.is_file():
        raise ValueError(f"No {description}; every dataset needs one.")
    folder = Path(data_root) / dataset if data_root else description.parent
    folder.mkdir(parents=True, exist_ok=True)
    return [fetch_one(entry, folder) for entry in parse_files_section(description)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset's declared data files if missing.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", help="where to put the files (default: data/ in this repo)")
    args = parser.parse_args()
    try:
        for status in fetch(args.dataset, args.data_root):
            print(status)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
