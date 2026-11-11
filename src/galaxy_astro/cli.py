"""Local fingerprinting and a synthetic demo. No RAW processing is automatic."""

import argparse
import json
from pathlib import Path

from .demo import synthetic_demo
from .provenance import snapshot, verify_snapshot, write_new_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="galaxy-astro")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("manifest", help="Read-only input fingerprints")
    create.add_argument("input", type=Path)
    create.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="Verify an exact input file set")
    verify.add_argument("input", type=Path)
    verify.add_argument("manifest", type=Path)
    demo = commands.add_parser("demo", help="Run known-truth synthetic checks")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            # Check the output before a potentially long input scan.
            if args.output.resolve().is_relative_to(args.input.resolve()):
                raise ValueError("Manifest must be saved outside the input directory")
            if args.output.exists():
                raise FileExistsError(args.output)
            report = snapshot(args.input)
            write_new_json(args.output, report, input_root=args.input)
            print(json.dumps(dict(files=len(report["files"]), written=str(args.output))))
            return 0
        if args.command == "verify":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            report = verify_snapshot(args.input, manifest)
        else:
            report = synthetic_demo(args.seed)
            if args.output is not None:
                write_new_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
