"""Write week-8 target-realism report and optional conservative MATE v2 targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.target_realism import write_target_realism_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", default="data/pilot/language_probe_v3_fix")
    parser.add_argument("--aux-root", default="data/pilot/language_probe_v4")
    parser.add_argument(
        "--mate-keyword-audit-path",
        default="data/pilot/language_probe_v3/reports/mate_keyword_audit.json",
    )
    parser.add_argument("--output-root", default="data/pilot/language_probe_v4")
    parser.add_argument(
        "--create-mate-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_target_realism_report(
        probe_root=args.probe_root,
        aux_root=args.aux_root,
        mate_keyword_audit_path=args.mate_keyword_audit_path,
        output_root=args.output_root,
        create_mate_v2=args.create_mate_v2,
    )
    print(f"Wrote target realism report to {result['json_path']}")


if __name__ == "__main__":
    main()
