"""Read-only input fingerprints and exclusive creation of JSON reports."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(root):
    """Fingerprint regular files; record relative paths, size, mtime and SHA256.

    Refuse symlinks and a file changing during its read. No EXIF or image
    decoding is performed. Store the returned report outside the input root.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Input root must be a directory")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not accepted in an input snapshot")
        if not path.is_file():
            continue
        before = path.stat()
        checksum = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("An input changed during fingerprinting")
        records.append(
            dict(
                path=path.relative_to(root).as_posix(),
                size=before.st_size,
                mtime_ns=before.st_mtime_ns,
                sha256=checksum,
            )
        )
    return dict(schema_version=1, algorithm="sha256", files=records)


def verify_snapshot(root, manifest):
    """Check the exact file set and all saved fields, without changing files."""
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise ValueError("Unsupported manifest schema")
    expected = {}
    for row in manifest["files"]:
        rel = row["path"]
        parsed = PurePosixPath(rel)
        if (
            not rel
            or parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in rel
            or ":" in rel
            or rel in expected
        ):
            raise ValueError("Unsafe or duplicate relative manifest path")
        expected[rel] = row
    actual = {row["path"]: row for row in snapshot(root)["files"]}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(p for p in expected.keys() & actual.keys() if expected[p] != actual[p])
    return dict(
        passed=not (missing or unexpected or changed),
        missing=missing,
        unexpected=unexpected,
        changed=changed,
        checked=len(actual),
    )


def write_new_json(path, data, *, input_root=None):
    """Create a new report, refusing overwrites and output inside input_root."""
    path = Path(path).resolve()
    if input_root is not None and path.is_relative_to(Path(input_root).resolve()):
        raise ValueError("Reports must be outside the original-input directory")
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
