"""コマンドライン。

  kirinuki run      動画/URL から切り抜きを作る
  kirinuki analyze  検出だけ（書き出さない）
  kirinuki review   出力動画を、伸びている切り抜きと比べてレビュー材料を作る
  kirinuki truth    既存の切り抜きが元配信のどこかを特定し、正解データを作る
  kirinuki eval     AIの選んだ区間を正解データで採点する
  kirinuki refs     伸びている切り抜きを yt-dlp で取得する
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Config


def _add_run_args(q: argparse.ArgumentParser) -> None:
    q.add_argument("video", help="動画ファイル または YouTube等のURL（yt-dlp が必要）")
    q.add_argument("-o", "--out", default="out")
    q.add_argument("-t", "--transcript", help=".srt / .vtt / .json（省略時は faster-whisper）")
    q.add_argument("-c", "--chat", help="チャットログ (yt-dlp live_chat.json / JSON / CSV)")
    q.add_argument("-p", "--profile", choices=["noripro", "hololive"], help="事務所ごとの規約チェック")
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kirinuki", description="最強切り抜きAI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    _add_run_args(sub.add_parser("run", help="見どころを検出して書き出す"))
    _add_run_args(sub.add_parser("analyze", help="検出のみ（書き出さない）"))

    q = sub.add_parser("review", help="出力動画のレビュー材料（コマ割り画像・指標・参照との差分）")
    q.add_argument("video")
    q.add_argument("--ref", nargs="*", default=[], help="比較する伸びている切り抜き動画")
    q.add_argument("--title")
    q.add_argument("-o", "--out", default="review")

    q = sub.add_parser("truth", help="既存の切り抜きの元区間を音声で特定し、正解データを作る")
    q.add_argument("stream", help="元配信の動画/音声")
    q.add_argument("clips", nargs="+", help="既存の切り抜き動画")
    q.add_argument("-o", "--out", default="truth.json")

    q = sub.add_parser("eval", help="report.json を正解データで採点")
    q.add_argument("report")
    q.add_argument("truth")

    q = sub.add_parser("refs", help="伸びている切り抜きを取得（動画＋再生数などのメタデータ）")
    q.add_argument("urls", nargs="+")
    q.add_argument("-o", "--out", default="refs")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if a.cmd in ("run", "analyze"):
        from .pipeline import run
        cfg = Config(
            num_clips=a.num, min_duration=a.min_duration, max_duration=a.max_duration,
            target_duration=a.target_duration, use_llm=not a.no_llm, model=a.model,
            vertical=not a.horizontal, teaser=a.teaser, font=a.font, min_relative_score=a.min_relative,
        )
        res = run(a.video, a.out, cfg, transcript=a.transcript, chat=a.chat, render_clips=a.cmd == "run",
                  thumbnails=not a.no_thumbnail, profile=a.profile,
                  whisper_model=a.whisper_model, language=a.language or None)
        for i, c in enumerate(res.clips, 1):
            print(f"#{i} {c.start:8.1f}s - {c.end:8.1f}s ({c.duration:4.1f}s) score={c.score:.3f} {c.title}")
        print(f"→ {a.out}/report.md")
        print("※ 投稿前に配信者の切り抜きガイドラインを必ず確認してください。")
    elif a.cmd == "review":
        from .review import review
        print(f"→ {review(a.video, a.ref, a.out, title=a.title)}")
    elif a.cmd == "truth":
        from .evaluate import build_truth
        truth = build_truth(a.stream, a.clips)
        Path(a.out).write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        for t in truth:
            where = f"{t['start']:.1f}〜{t['end']:.1f}s (確信度 {t['confidence']})" if "start" in t else "見つからず"
            print(f"{Path(t['file']).name}: {where}")
        print(f"→ {a.out}")
    elif a.cmd == "eval":
        from .evaluate import score
        rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
        truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
        print(json.dumps(score(rep["clips"], truth), ensure_ascii=False, indent=2))
    elif a.cmd == "refs":
        from .source import download_refs
        for f in download_refs(a.urls, a.out):
            print(f)
    return 0
