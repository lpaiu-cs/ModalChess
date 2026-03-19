"""로깅 유틸리티."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """간단한 스트림 로거를 생성한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
