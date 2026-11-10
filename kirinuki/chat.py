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

from .audio import local_jump
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


def chat_signal(msgs: list[Message], n_sec: int, delay: float = 5.0) -> np.ndarray:
    """1秒ごとのチャット盛り上がり度 (0〜1)。コメント量の跳ね × 反応語の濃さ。"""
    if not msgs or n_sec <= 0:
        return np.zeros(max(n_sec, 0))
    count = np.zeros(n_sec)
    react = np.zeros(n_sec)
    for t, text in msgs:
        i = int(math.floor(t - delay))
        if 0 <= i < n_sec:
            count[i] += 1
            react[i] += reaction_score(text)
    # 3秒窓で平滑化（反応のばらつきを吸収）
    k = np.ones(3) / 3
    count_s = np.convolve(count, k, mode="same")
    react_s = np.convolve(react, k, mode="same")
    rate = local_jump(np.log1p(count_s), window=120, min_spread=0.25, ignore_below=None)
    density = np.where(count_s > 0, react_s / np.maximum(count_s, 1e-6), 0.0)
    return np.clip(0.6 * rate + 0.4 * density * (rate > 0), 0.0, 1.0)
