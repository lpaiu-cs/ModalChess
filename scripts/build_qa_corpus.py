"""P0: D1 포지션에서 grounded QA 코퍼스 생성 + 독립 검증 + 위생/균형 리포트.

split 매핑(docs/phase2_plan.md §5): QA-train←supervised_train(인코더 노출 허용),
QA-val←supervised_val, QA-test←supervised_test(인코더 미노출). 크로스-split 포지션
중복은 test > val > train 우선순위로 제거. test만 held-out 템플릿 포함.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.fusion.qa_generator import generate_split  # noqa: E402
from modalchess.fusion.qa_tasks import position_key  # noqa: E402
from modalchess.fusion.qa_verifier import verify_corpus  # noqa: E402


def load_positions(path: Path) -> tuple[list[tuple[str, str]], set[str], set[str]]:
    """(fen, game_id) 리스트(포지션 키 중복 제거) + 키 집합 + game_id 집합."""
    positions: list[tuple[str, str]] = []
    keys: set[str] = set()
    games: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            fen = row["fen"]
            key = position_key(fen)
            games.add(str(row["game_id"]))
            if key in keys:
                continue
            keys.add(key)
            positions.append((fen, str(row["game_id"])))
    return positions, keys, games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pilot/real_v2_scale")
    parser.add_argument("--out-dir", default="outputs/phase2/qa_v1")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--train-scale", type=float, default=1.0)
    parser.add_argument("--val-scale", type=float, default=0.05)
    parser.add_argument("--test-scale", type=float, default=0.10)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading positions...", flush=True)
    test_pos, test_keys, test_games = load_positions(data_root / "supervised_test.jsonl")
    val_pos, val_keys, val_games = load_positions(data_root / "supervised_val.jsonl")
    train_pos, train_keys, train_games = load_positions(data_root / "supervised_train.jsonl")

    # 크로스-split 포지션 중복 제거 (test > val > train)
    val_pos = [(f, g) for f, g in val_pos if position_key(f) not in test_keys]
    blocked = test_keys | val_keys
    train_pos = [(f, g) for f, g in train_pos if position_key(f) not in blocked]

    hygiene = {
        "unique_positions": {"train": len(train_pos), "val": len(val_pos), "test": len(test_pos)},
        "game_overlap": {
            "train∩val": len(train_games & val_games),
            "train∩test": len(train_games & test_games),
            "val∩test": len(val_games & test_games),
        },
    }
    print("hygiene:", json.dumps(hygiene, ensure_ascii=False), flush=True)

    stats: dict[str, object] = {"hygiene": hygiene, "splits": {}}
    specs = [
        ("train", train_pos, args.train_scale, False),
        ("val", val_pos, args.val_scale, False),
        ("test", test_pos, args.test_scale, True),
    ]
    total_mismatched = 0
    for name, pool, scale, with_held_out in specs:
        print(f"generating {name} (scale={scale})...", flush=True)
        items, shortfall = generate_split(
            pool, quota_scale=scale, seed=args.seed + len(name),
            include_held_out_template=with_held_out,
        )
        for item in items:
            item["split"] = name
        report = verify_corpus(items)
        total_mismatched += report["n_mismatched"]
        out_path = out_dir / f"qa_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        n_held_out = sum(1 for i in items if i["template_id"] == 3)
        stats["splits"][name] = {
            "n_items": report["n_items"],
            "n_mismatched": report["n_mismatched"],
            "error_samples": report["error_samples"],
            "answer_distribution": report["answer_distribution"],
            "quota_shortfall": shortfall,
            "n_held_out_template_items": n_held_out,
            "n_unique_positions_used": len({i["position_key"] for i in items}),
        }
        print(f"  {name}: {report['n_items']} items, mismatched={report['n_mismatched']}, "
              f"shortfall_tasks={list(shortfall)}, held_out_tpl={n_held_out}", flush=True)

    (out_dir / "qa_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"P0_CORPUS_DONE mismatched_total={total_mismatched}", flush=True)


if __name__ == "__main__":
    main()
