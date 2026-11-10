"""共通データ型。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    """文字起こしの1発話（おおむね1文）。"""
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Clip:
    """切り抜き候補 / 確定クリップ。時刻はすべて元動画の秒。"""
    start: float
    end: float
    peak: float
    score: float = 0.0
    features: dict = field(default_factory=dict)
    seg_range: tuple[int, int] = (0, 0)  # 含まれるセグメントの [first, last]
    title: str = ""
    hook_text: str = ""
    keywords: list[str] = field(default_factory=list)
    reason: str = ""
    llm_scores: Optional[dict] = None
    teaser: Optional[tuple[float, float]] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d
