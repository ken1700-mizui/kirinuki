"""チャット（コメント）シグナル。

人間の切り抜き師が最初に見るのは「コメントが急に流れた所」と「草が並んだ所」。
視聴者は出来事の数秒後に反応するので、遅延分だけ前にずらして使う。

対応形式:
- yt-dlp の YouTube ライブチャット (`--write-subs --sub-langs live_chat` で得られる *.live_chat.json, JSON Lines)
- 汎用 JSON: [{"time": 秒, "text": "..."}]  （"offset" / "content_offset_seconds" なども可）
- CSV: time,text
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .text import reaction_score

Message = tuple[float, str]


def _yt_runs(msg: dict) -> str:
    return "".join(r.get("text") or r.get("emoji", {}).get("shortcuts", [""])[0] for r in msg.get("runs", []))


def _parse_ytdlp_line(obj: dict) -> list[Message]:
    rep = obj.get("replayChatItemAction")
    if not rep:
        return []
    t = float(rep.get("videoOffsetTimeMsec", obj.get("videoOffsetTimeMsec", 0))) / 1000
    out = []
    for act in rep.get("actions", []):
        item = act.get("addChatItemAction", {}).get("item", {})
        for key in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer"):
            r = item.get(key)
            if r and "message" in r:
                out.append((t, _yt_runs(r["message"])))
    return out


def load_chat(path: str | Path) -> list[Message]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8-sig")
    msgs: list[Message] = []
    if p.suffix.lower() == ".csv":
        for row in csv.reader(raw.splitlines()):
            try:
                msgs.append((float(row[0]), row[1] if len(row) > 1 else ""))
            except (ValueError, IndexError):
                continue  # ヘッダ行など
        return sorted(msgs)

    stripped = raw.lstrip()
    if stripped.startswith("["):
        for m in json.loads(raw):
            t = m.get("time", m.get("offset", m.get("content_offset_seconds")))
            if t is not None:
                msgs.append((float(t), str(m.get("text", m.get("message", "")))))
        return sorted(msgs)

    for line in raw.splitlines():  # JSON Lines (yt-dlp)
        line = line.strip()
        if line:
            try:
                msgs.extend(_parse_ytdlp_line(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return sorted(msgs)


def _rolling_quantile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    n, half = len(x), window // 2
    return np.array([np.percentile(x[max(0, t - half):min(n, t + half + 1)], q) for t in range(n)])


def chat_signal(msgs: list[Message], n_sec: int, delay: float = 5.0, window: int = 300,
                full_ratio: float = 8.0) -> np.ndarray:
    """1秒ごとのチャット盛り上がり度 (0〜1)。

    - 量: 「普段(前後 window 秒の40%点)の何倍の速さでコメントが流れているか」。full_ratio 倍で満点。
      盛り上がりが多い配信でも基準が引っ張られにくいよう、中央値ではなく低めの分位点を基準にする。
    - 質: 反応語（草・w・えぐい 等）を含むコメントの割合。
    """
    if not msgs or n_sec <= 0:
        return np.zeros(max(n_sec, 0))
    count = np.zeros(n_sec)
    react = np.zeros(n_sec)
    for t, text in msgs:
        i = int(math.floor(t - delay))
        if 0 <= i < n_sec:
            count[i] += 1
            react[i] += min(1.0, reaction_score(text) * 2.5)  # 「草」1語でも立派な反応
    k = np.ones(3) / 3
    count_s = np.convolve(count, k, mode="same")
    react_s = np.convolve(react, k, mode="same")
    base = _rolling_quantile(count_s, window, 40)
    ratio = (count_s + 0.5) / (base + 0.5)
    rate = np.clip(np.log2(np.maximum(ratio, 1.0)) / math.log2(full_ratio), 0.0, 1.0)
    quality = np.where(count_s > 0, react_s / np.maximum(count_s, 1e-6), 0.0)
    return np.clip(rate * (0.6 + 0.4 * quality), 0.0, 1.0)
