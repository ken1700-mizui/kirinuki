"""URL 入力: yt-dlp で動画・ライブチャット・字幕をまとめて取得する。

権利の確認は利用者の責任。取得前に配信者の切り抜きガイドラインを確認すること。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    video: Path
    chat: Path | None = None
    subtitles: Path | None = None


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def ytdlp_cmd(url: str, dest: Path, sub_lang: str = "ja") -> list[str]:
    return [
        "yt-dlp", url,
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "--write-subs", "--sub-langs", f"live_chat,{sub_lang}", "--sub-format", "json/vtt/srt/best",
        "-o", str(dest / "source.%(ext)s"),
        "--no-playlist",
    ]


def find_outputs(dest: Path) -> Source:
    video = next((p for p in sorted(dest.glob("source.*")) if p.suffix in (".mp4", ".mkv", ".webm")), None)
    if video is None:
        raise RuntimeError("yt-dlp の出力動画が見つかりません")
    chat = next(iter(sorted(dest.glob("source*.live_chat.json"))), None)
    subs = next((p for p in sorted(dest.glob("source*.vtt")) if "live_chat" not in p.name), None)
    if subs is None:
        subs = next(iter(sorted(dest.glob("source*.srt"))), None)
    return Source(video, chat, subs)


def download(url: str, dest: str | Path, sub_lang: str = "ja") -> Source:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp が必要です: pip install yt-dlp")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(ytdlp_cmd(url, dest, sub_lang), capture_output=True, text=True)
    if r.returncode != 0:
        # ライブチャットが無いだけなら動画は取れている
        try:
            return find_outputs(dest)
        except RuntimeError:
            raise RuntimeError(f"ダウンロード失敗:\n{r.stderr[-1500:]}") from None
    return find_outputs(dest)
