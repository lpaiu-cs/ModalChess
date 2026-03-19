"""YAML 설정 로딩 유틸리티."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 딕셔너리로 읽는다."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """중첩 딕셔너리를 재귀적으로 병합한다."""
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_and_merge_yaml_configs(paths: list[str | Path]) -> dict[str, Any]:
    """여러 YAML 파일을 순서대로 읽어 병합한다."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge_dict(merged, load_yaml_config(path))
    return merged
