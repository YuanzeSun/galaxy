import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from galaxy_astro.cli import main
from galaxy_astro.demo import synthetic_demo
from galaxy_astro.provenance import snapshot, verify_snapshot, write_new_json


class ProvenanceTests(unittest.TestCase):
    def test_manifest_read_only_and_change_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "inputs"
            root.mkdir()
            file = root / "sample.bin"
            file.write_bytes(b"original data")
            before = file.stat()
            manifest = snapshot(root)
            self.assertTrue(verify_snapshot(root, manifest)["passed"])
            self.assertEqual(file.read_bytes(), b"original data")
            self.assertEqual(file.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(manifest["files"][0]["path"], "sample.bin")
            file.write_bytes(b"changed data!")
            self.assertEqual(verify_snapshot(root, manifest)["changed"], ["sample.bin"])
            file.unlink()
            (root / "other.bin").write_bytes(b"new")
            result = verify_snapshot(root, manifest)
            self.assertEqual(result["missing"], ["sample.bin"])
            self.assertEqual(result["unexpected"], ["other.bin"])

    def test_output_cannot_overwrite_or_enter_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "inputs"
            root.mkdir()
            output = Path(d) / "manifest.json"
            write_new_json(output, {"x": 1}, input_root=root)
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"x": 2})
            with self.assertRaises(ValueError):
                write_new_json(root / "report.json", {}, input_root=root)
            self.assertEqual(json.loads(output.read_text()), {"x": 1})

    def test_untrusted_manifest_paths(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ["../bad", "/absolute", "X:/absolute", "a\\b"]:
                data = dict(schema_version=1, algorithm="sha256", files=[dict(path=path)])
                with self.assertRaises(ValueError):
                    verify_snapshot(d, data)

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d) / "inputs"
            root.mkdir()
            (root / "a").write_bytes(b"data")
            output = Path(d) / "manifest.json"
            self.assertEqual(main(["manifest", str(root), "--output", str(output)]), 0)
            self.assertEqual(main(["verify", str(root), str(output)]), 0)
            with self.assertRaises(SystemExit) as exc:
                with contextlib.redirect_stderr(io.StringIO()):
                    main(["manifest", str(root), "--output", str(root / "bad.json")])
            self.assertEqual(exc.exception.code, 2)
            self.assertFalse((root / "bad.json").exists())

    def test_synthetic_demo(self):
        report = synthetic_demo()
        self.assertTrue(report["synthetic_only"])
        self.assertTrue(report["passed"], report)
