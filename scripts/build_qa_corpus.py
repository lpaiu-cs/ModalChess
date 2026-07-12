"""P0: D1 포지션에서 grounded QA 코퍼스 생성 + 독립 검증 + 위생/균형 리포트.

split 매핑(docs/phase2_plan.md §5): QA-train←supervised_train(인코더 노출 허용),
QA-val←supervised_val, QA-test←supervised_test(인코더 미노출). 크로스-split 포지션
중복은 test > val > train 우선순위로 제거. test만 held-out 템플릿 포함.

코퍼스 버전 재현(--tiers로 task-set 명시):
  qa_v1: python scripts/build_qa_corpus.py --out-dir outputs/phase2/qa_v1 --tiers T1,T2
  qa_v2: python scripts/build_qa_corpus.py --out-dir outputs/phase2/qa_v2 --tiers T1,T2,T3
검증 불일치 또는 쿼터 미충족이 있으면 아티팩트 발행 전에 exit 1로 실패한다(P0 규율).
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
    # 코퍼스 버전별 task-set 명시(재현성): qa_v1=T1,T2 (기본) / qa_v2=T1,T2,T3
    parser.add_argument("--tiers", default="T1,T2",
                        help="생성할 tier CSV. 기본 'T1,T2'(qa_v1). qa_v2는 'T1,T2,T3'.")
    args = parser.parse_args()
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading positions...", flush=True)
    test_pos, test_keys, test_games = load_positions(data_root / "supervised_test.jsonl")
    val_pos, val_keys, val_games = load_positions(data_root / "supervised_val.jsonl")
    train_pos, train_keys, train_games = load_positions(data_root / "supervised_train.jsonl")

    raw_game_overlap = {
        "train∩val": len(train_games & val_games),
        "train∩test": len(train_games & test_games),
        "val∩test": len(val_games & test_games),
    }
    # 위생 강제(phase2_plan §5, test > val > train): 포지션 dedup에 더해 **game 단위** 홀드아웃.
    # 상위 split의 game_id를 하위 split에서 통째로 제거 → QA-train이 QA-test 게임의 인접
    # 포지션을 포함하지 못하게(단순 position dedup으로는 못 막는 오염 차단).
    val_pos = [(f, g) for f, g in val_pos
               if g not in test_games and position_key(f) not in test_keys]
    blocked_keys = test_keys | val_keys
    blocked_games = test_games | val_games
    train_pos = [(f, g) for f, g in train_pos
                 if g not in blocked_games and position_key(f) not in blocked_keys]

    # 제거 후 game 교차가 0이어야 함(강제의 사후 검증). 아니면 실패.
    post_train_games = {g for _, g in train_pos}
    post_val_games = {g for _, g in val_pos}
    residual = (len(post_train_games & post_val_games)
                + len(post_train_games & test_games) + len(post_val_games & test_games))
    hygiene = {
        "unique_positions": {"train": len(train_pos), "val": len(val_pos), "test": len(test_pos)},
        "raw_game_overlap": raw_game_overlap,
        "residual_game_overlap_after_holdout": residual,
    }
    print("hygiene:", json.dumps(hygiene, ensure_ascii=False), flush=True)
    if residual > 0:
        print(f"P0_CORPUS_FAIL residual game overlap after holdout: {residual}", flush=True)
        raise SystemExit(1)

    stats: dict[str, object] = {"hygiene": hygiene, "tiers": list(tiers), "splits": {}}
    specs = [
        ("train", train_pos, args.train_scale, False),
        ("val", val_pos, args.val_scale, False),
        ("test", test_pos, args.test_scale, True),
    ]
    total_mismatched = 0
    for name, pool, scale, with_held_out in specs:
        print(f"generating {name} (scale={scale}, tiers={tiers})...", flush=True)
        items, shortfall = generate_split(
            pool, quota_scale=scale, seed=args.seed + len(name),
            include_held_out_template=with_held_out, tiers=tiers,
        )
        for item in items:
            item["split"] = name
        report = verify_corpus(items)
        total_mismatched += report["n_mismatched"]
        # P0 규율(phase2_plan §6): 아티팩트 발행 전에 즉시 실패해야 하는 두 조건.
        # (1) 검증기 불일치 — bad 생성기/검증기가 조용히 유효 코퍼스로 둔갑하는 것 차단.
        if report["n_mismatched"] > 0:
            print(f"P0_CORPUS_FAIL split={name} mismatched={report['n_mismatched']} "
                  f"samples={report['error_samples'][:5]}", flush=True)
            raise SystemExit(1)
        # (2) 쿼터 미충족 — P0 balance 보장이 깨진 불균형 코퍼스로 하류 비교가 진행되는 것 차단.
        if shortfall:
            print(f"P0_CORPUS_FAIL split={name} quota_shortfall={shortfall}", flush=True)
            raise SystemExit(1)
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
