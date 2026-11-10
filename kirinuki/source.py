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
    info: Path | None = None


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _auth(cookies: str | None, cookies_from_browser: str | None) -> list[str]:
    """自分のアカウントでの認証（ボット判定を避けるための正規の方法）。"""
    if cookies:
        return ["--cookies", cookies]
    if cookies_from_browser:
        return ["--cookies-from-browser", cookies_from_browser]
    return []


def ytdlp_cmd(url: str, dest: Path, sub_lang: str = "ja", cookies: str | None = None,
              cookies_from_browser: str | None = None) -> list[str]:
    return [
        "yt-dlp", url, *_auth(cookies, cookies_from_browser),
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        # 手動字幕 → 無ければ自動字幕(json3 は単語ごとの時刻つき) + ライブチャット
        "--write-subs", "--write-auto-subs", "--sub-langs", f"live_chat,{sub_lang},{sub_lang}-orig",
        "--sub-format", "json3/json/vtt/srt/best",
        "-o", str(dest / "source.%(ext)s"),
        "--write-info-json",
        "--no-playlist",
    ]


def ytdlp_ref_cmd(url: str, dest: Path, cookies: str | None = None,
                  cookies_from_browser: str | None = None) -> list[str]:
    """参照用（伸びている切り抜き）: 動画＋メタデータ(再生数・タイトル)。"""
    return ["yt-dlp", url, *_auth(cookies, cookies_from_browser), "-f", "bv*[height<=720]+ba/b", "--merge-output-format", "mp4",
            "--write-info-json", "-o", str(dest / "%(id)s.%(ext)s"), "--no-playlist"]


def find_outputs(dest: Path) -> Source:
    video = next((p for p in sorted(dest.glob("source.*")) if p.suffix in (".mp4", ".mkv", ".webm")), None)
    if video is None:
        raise RuntimeError("yt-dlp の出力動画が見つかりません")
    chat = next(iter(sorted(dest.glob("source*.live_chat.json"))), None)
    subs = next(iter(sorted(dest.glob("source*.json3"))), None)
    if subs is None:
        subs = next((p for p in sorted(dest.glob("source*.vtt")) if "live_chat" not in p.name), None)
    if subs is None:
        subs = next(iter(sorted(dest.glob("source*.srt"))), None)
    info = next(iter(sorted(dest.glob("source.info.json"))), None)
    return Source(video, chat, subs, info)


def download(url: str, dest: str | Path, sub_lang: str = "ja", cookies: str | None = None,
             cookies_from_browser: str | None = None) -> Source:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp が必要です: pip install yt-dlp")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(ytdlp_cmd(url, dest, sub_lang, cookies, cookies_from_browser), capture_output=True, text=True)
    if r.returncode != 0:
        # ライブチャットが無いだけなら動画は取れている
        try:
            return find_outputs(dest)
        except RuntimeError:
            raise RuntimeError(f"ダウンロード失敗:\n{r.stderr[-1500:]}") from None
    return find_outputs(dest)


def download_refs(urls: list[str], dest: str | Path, cookies: str | None = None,
                  cookies_from_browser: str | None = None) -> list[Path]:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp が必要です: pip install yt-dlp")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for u in urls:
        subprocess.run(ytdlp_ref_cmd(u, dest, cookies, cookies_from_browser), capture_output=True, text=True)
    for p in sorted(dest.glob("*.mp4")):
        out.append(p)
    return out


def load_info(video: str | Path) -> dict:
    """動画の隣にある yt-dlp の info.json を読む（無ければ空）。"""
    import json
    v = Path(video)
    for cand in (v.with_suffix(".info.json"), v.parent / "source.info.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except ValueError:
                return {}
    return {}
