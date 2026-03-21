"""Audit auxiliary language sources for week-7 without changing source semantics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.aux_language import audit_aux_language_sources


def _parse_source_args(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must use name=path form: {value}")
        name, path = value.split("=", 1)
        parsed[name] = path
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pilot/language_probe_v3/reports")
    parser.add_argument("--source", action="append", default=[], help="name=path form; repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_aux_language_sources(
        source_paths=_parse_source_args(args.source) if args.source else None,
        output_dir=args.output_dir,
    )
    print(f"Wrote auxiliary source audit to {result['json_path']}")


if __name__ == "__main__":
    main()
