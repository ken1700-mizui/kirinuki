"""コマンドライン: python -m kirinuki run 動画.mp4 -o out/"""
from __future__ import annotations

import argparse
import logging

from .config import Config
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kirinuki", description="最強切り抜きAI")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_ in (("run", "見どころを検出して書き出す"), ("analyze", "検出のみ（書き出さない）")):
        q = sub.add_parser(name, help=help_)
        q.add_argument("video", help="動画ファイル または YouTube等のURL（yt-dlp が必要）")
        q.add_argument("-o", "--out", default="out")
        q.add_argument("-t", "--transcript", help=".srt / .vtt / .json（省略時は faster-whisper）")
        q.add_argument("-c", "--chat", help="チャットログ (yt-dlp live_chat.json / JSON / CSV)")
        q.add_argument("-n", "--num", type=int, default=5, help="クリップ数")
        q.add_argument("--min", type=float, default=15.0, dest="min_duration")
        q.add_argument("--max", type=float, default=60.0, dest="max_duration")
        q.add_argument("--target", type=float, default=30.0, dest="target_duration")
        q.add_argument("--min-relative", type=float, default=0.5, help="1位の何割未満のクリップは出さない")
        q.add_argument("--no-llm", action="store_true", help="Claude を使わない")
        q.add_argument("--model", default="claude-opus-5-5")
        q.add_argument("--horizontal", action="store_true", help="横型(16:9)で出力")
        q.add_argument("--teaser", choices=["auto", "on", "off"], default="auto")
        q.add_argument("--font", default="IPAGothic")
        q.add_argument("--whisper-model", default="large-v3")
        q.add_argument("--language", default="ja")
        q.add_argument("--no-thumbnail", action="store_true")
        q.add_argument("-v", "--verbose", action="store_true")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = Config(
        num_clips=a.num, min_duration=a.min_duration, max_duration=a.max_duration,
        target_duration=a.target_duration, use_llm=not a.no_llm, model=a.model,
        vertical=not a.horizontal, teaser=a.teaser, font=a.font, min_relative_score=a.min_relative,
    )
    res = run(a.video, a.out, cfg, transcript=a.transcript, chat=a.chat, render_clips=a.cmd == "run", thumbnails=not a.no_thumbnail,
              whisper_model=a.whisper_model, language=a.language or None)
    for i, c in enumerate(res.clips, 1):
        print(f"#{i} {c.start:8.1f}s - {c.end:8.1f}s ({c.duration:4.1f}s) score={c.score:.3f} {c.title}")
    print(f"→ {a.out}/report.md")
    print("※ 投稿前に配信者の切り抜きガイドラインを必ず確認してください。")
    return 0
