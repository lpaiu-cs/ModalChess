"""Audit how realistic current week-6 probes are with respect to raw text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.text_realism_audit import audit_probe_text_realism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/language_probe_v2")
    parser.add_argument("--week6-readiness-path", default="outputs/week6/readiness_probes/probe_results.json")
    parser.add_argument("--week6-retrieval-path", default="outputs/week6/retrieval_probes/retrieval_results.json")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_probe_text_realism(
        input_root=args.input_root,
        week6_readiness_path=args.week6_readiness_path,
        week6_retrieval_path=args.week6_retrieval_path,
        output_dir=args.output_dir,
    )
    print(f"Wrote text realism audit to {result['json_path']}")


if __name__ == "__main__":
    main()
