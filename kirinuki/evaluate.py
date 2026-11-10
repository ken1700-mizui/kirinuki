"""精度評価: AIが選んだ区間が、人間の「伸びた切り抜き」とどれだけ一致するか。

正解データ (truth.json):
  [{"start": 秒, "end": 秒, "views": 再生数, "title": "...", "file": "..."}]
  → `kirinuki locate` で既存の切り抜き動画から自動生成できる。
"""
from __future__ import annotations

import json
from pathlib import Path

from .locate import locate


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def build_truth(stream: str | Path, clips: list[str | Path], min_conf: float = 0.3) -> list[dict]:
    from .audio import analyze_audio
    sa = analyze_audio(stream)
    truth = []
    for c in clips:
        m = locate(c, sa)
        item = {"file": str(c), "found": bool(m and m.confidence >= min_conf)}
        meta = Path(c).with_suffix(".info.json")
        if meta.exists():
            d = json.loads(meta.read_text(encoding="utf-8"))
            item.update(views=d.get("view_count"), title=d.get("title"))
        if m:
            item.update(start=round(m.start, 2), end=round(m.end, 2), confidence=m.confidence)
        truth.append(item)
    return truth


def score(pred: list[dict], truth: list[dict], thresh: float = 0.5) -> dict:
    """pred: report.json の clips（順位順）, truth: build_truth の出力（found のみ使う）。"""
    truth = [t for t in truth if t.get("found", True) and "start" in t]
    if not truth:
        return {"error": "正解データがありません"}
    total_views = sum(t.get("views") or 1 for t in truth)

    def hit(p, t):
        return _overlap((p["start"], p["end"]), (t["start"], t["end"])) >= thresh

    res = {"n_truth": len(truth), "n_pred": len(pred)}
    for k in (1, 3, 5, 10):
        top = pred[:k]
        covered = [t for t in truth if any(hit(p, t) for p in top)]
        res[f"recall@{k}"] = round(len(covered) / len(truth), 3)
        res[f"view_recall@{k}"] = round(sum(t.get("views") or 1 for t in covered) / total_views, 3)
    res["precision"] = round(sum(any(hit(p, t) for t in truth) for p in pred) / max(1, len(pred)), 3)
    # どの正解を逃したか（改善のヒント）
    res["missed"] = [{"start": t["start"], "end": t["end"], "views": t.get("views"), "title": t.get("title")}
                     for t in truth if not any(hit(p, t) for p in pred)]
    return res
