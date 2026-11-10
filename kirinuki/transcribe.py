"""文字起こしの読み込み / 生成。

- SRT / VTT ファイル
- JSON（whisper / faster-whisper 互換: {"segments": [{start, end, text, words: [{start, end, word}]}]}）
- faster-whisper による自動文字起こし（インストールされていてモデルを取得できる場合）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Segment, Word

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")


def _ts_to_sec(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"不正なタイムスタンプ: {s!r}")
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000


def load_srt(path: str | Path) -> list[Segment]:
    """SRT / WebVTT を読み込む。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    segs: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")
        body = " ".join(lines[idx + 1:])
        body = re.sub(r"<[^>]+>", "", body).strip()  # VTT/HTML タグ除去
        if body:
            segs.append(Segment(_ts_to_sec(a), _ts_to_sec(b), body))
    return normalize(segs)


def load_json(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["segments"] if isinstance(data, dict) else data
    segs = []
    for s in raw:
        words = [
            Word(float(w["start"]), float(w["end"]), str(w.get("word", w.get("text", ""))))
            for w in s.get("words") or []
            if w.get("start") is not None and w.get("end") is not None
        ]
        segs.append(Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip(), words))
    return normalize(segs)


def save_json(segs: list[Segment], path: str | Path) -> None:
    out = {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.text} for w in s.words],
            }
            for s in segs
        ]
    }
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def load_transcript(path: str | Path) -> list[Segment]:
    p = Path(path)
    if p.suffix.lower() in (".srt", ".vtt"):
        return load_srt(p)
    if p.suffix.lower() == ".json":
        return load_json(p)
    raise ValueError(f"未対応の文字起こし形式: {p.suffix}（.srt / .vtt / .json）")


def transcribe_whisper(video: str | Path, model_size: str = "large-v3", language: str | None = "ja") -> list[Segment]:
    """faster-whisper で単語タイムスタンプ付き文字起こし。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("faster-whisper が未インストールです: pip install faster-whisper") from e

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    it, _info = model.transcribe(
        str(video),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # 長尺での幻覚ループ対策
    )
    segs = []
    for s in it:
        words = [Word(w.start, w.end, w.word) for w in (s.words or [])]
        segs.append(Segment(s.start, s.end, s.text.strip(), words))
    return normalize(segs)


def normalize(segs: list[Segment]) -> list[Segment]:
    """時刻順に並べ、空・逆転・重複を整える。"""
    segs = sorted((s for s in segs if s.text.strip() and s.end > s.start), key=lambda s: s.start)
    for prev, cur in zip(segs, segs[1:]):
        if cur.start < prev.end:  # 重なりは中点で分ける
            mid = (prev.end + cur.start) / 2
            prev.end = max(prev.start + 0.01, mid)
            cur.start = min(cur.end - 0.01, mid)
    return segs
