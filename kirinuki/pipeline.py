"""全体パイプライン: 解析 → 候補 → (LLM) → 選定 → 書き出し → レポート。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import llm
from .audio import analyze_audio, probe
from .chat import chat_signal, load_chat
from .config import Config
from .edit import plan_teaser, render, render_thumbnail
from .highlight import Timeline, build_timeline, generate_candidates, select
from .models import Clip, Segment
from . import profiles
from .source import download, is_url, load_info
from .text import reaction_words
from .transcribe import load_transcript, save_json, transcribe_whisper

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    info: dict
    segs: list[Segment]
    timeline: Timeline
    clips: list[Clip]
    used_llm: bool
    rule_warnings: list[str] | None = None


def auto_title(clip: Clip, segs: list[Segment]) -> str:
    """LLM なしのタイトル: ピーク発話を短く。"""
    i0, i1 = clip.seg_range
    window = segs[i0:i1 + 1] if segs else []
    if not window:
        return ""
    peak_seg = min(window, key=lambda s: 0 if s.start <= clip.peak <= s.end else abs(s.start - clip.peak))
    t = peak_seg.text.strip().rstrip("。、")
    return t if len(t) <= 18 else t[:17] + "…"


def analyze(video: str | Path, cfg: Config, transcript: str | Path | None = None,
            chat: str | Path | None = None, whisper_model: str = "large-v3", language: str | None = "ja", client=None) -> Analysis:
    info = probe(video)
    if transcript:
        segs = load_transcript(transcript)
    else:
        try:
            segs = transcribe_whisper(video, whisper_model, language)
        except Exception as e:
            log.warning("文字起こしに失敗（音声シグナルのみで続行）: %s", e)
            segs = []
    audio = analyze_audio(video)
    duration = info["duration"] or audio.duration
    chat_arr = None
    if chat:
        chat_arr = chat_signal(load_chat(chat), int(duration) + 1, cfg.chat_delay)
    tl = build_timeline(segs, audio, duration, cfg, chat_arr)
    cands = generate_candidates(segs, tl, duration, cfg)

    used_llm = False
    if cfg.use_llm and (client is not None or llm.available()):
        before = [id(c) for c in cands]
        cands = llm.rerank(cands, segs, duration, cfg, client=client)
        used_llm = any(c.llm_scores for c in cands) or [id(c) for c in cands] != before

    clips = select(cands, cfg.num_clips, cfg.overlap_threshold, cfg.min_relative_score)
    for c in clips:
        if not c.title:
            c.title = auto_title(c, segs)
        if not c.keywords and segs:
            text = "".join(s.text for s in segs[c.seg_range[0]:c.seg_range[1] + 1])
            c.keywords = reaction_words(text)[:3]
        c.teaser = plan_teaser(c, segs, cfg)
    return Analysis(info, segs, tl, clips, used_llm)


def run(video: str | Path, out_dir: str | Path, cfg: Config, transcript: str | Path | None = None,
        render_clips: bool = True, chat: str | Path | None = None, thumbnails: bool = True,
        profile: str | None = None, cookies: str | None = None,
        cookies_from_browser: str | None = None, **kw) -> Analysis:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = profiles.get(profile)
    if isinstance(video, str) and is_url(video):
        src = download(video, out_dir / "source", cookies=cookies, cookies_from_browser=cookies_from_browser)
        log.info("取得: %s (chat=%s, subs=%s)", src.video, src.chat, src.subtitles)
        video, chat = src.video, chat or src.chat
        # 配信者が付けた字幕は Whisper より正確なことが多い
        transcript = transcript or src.subtitles
    info = load_info(video)

    rule_warnings: list[str] = []
    if prof:
        errors, rule_warnings = profiles.check_source(prof, info)
        if errors:
            raise RuntimeError(f"{prof.name}の規約上、この配信は切り抜けません:\n- " + "\n- ".join(errors))

    a = analyze(video, cfg, transcript, chat=chat, **kw)
    a.rule_warnings = rule_warnings
    if a.segs and not transcript:
        save_json(a.segs, out_dir / "transcript.json")

    files: list[dict] = []
    for i, c in enumerate(a.clips, 1):
        name = f"clip_{i:02d}"
        c.warnings = profiles.check_title(c.title)
        (out_dir / f"{name}.txt").write_text(profiles.make_description(prof, info, c), encoding="utf-8")
        if not render_clips:
            continue
        entry = {"video": str(render(video, c, a.segs, a.timeline.silences, a.info, out_dir, name, cfg))}
        if thumbnails and a.info["has_video"]:
            try:
                entry["thumbnail"] = str(render_thumbnail(video, c, a.info, out_dir, name, cfg))
            except RuntimeError as e:
                log.warning("%s", e)
        files.append(entry)
    write_report(a, out_dir, files, str(video), prof)
    return a


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def write_report(a: Analysis, out_dir: Path, files: list[dict], src: str,
                 prof: "profiles.Profile | None" = None) -> None:
    data = {
        "source": src,
        "duration": a.info["duration"],
        "used_llm": a.used_llm,
        "profile": prof.key if prof else None,
        "rule_warnings": a.rule_warnings or [],
        "clips": [dict(c.to_dict(), file=(files[i] if i < len(files) else {}).get("video"),
                       thumbnail=(files[i] if i < len(files) else {}).get("thumbnail"))
                  for i, c in enumerate(a.clips)],
    }
    (out_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# 切り抜きレポート\n", f"- 元動画: `{src}`", f"- 長さ: {_fmt(a.info['duration'])}",
          f"- LLM再ランク: {'あり' if a.used_llm else 'なし（ヒューリスティックのみ）'}\n",
          "| # | 区間 | 尺 | スコア | タイトル | フック | ティーザー | 理由 |",
          "|---|---|---|---|---|---|---|---|"]
    esc = lambda x: (x or "-").replace("|", "｜")  # noqa: E731
    for i, c in enumerate(a.clips, 1):
        teaser = f"{_fmt(c.teaser[0])}〜" if c.teaser else "-"
        md.append(f"| {i} | {_fmt(c.start)}〜{_fmt(c.end)} | {c.duration:.0f}s | {c.score:.3f} | "
                  f"{esc(c.title)} | {esc(c.hook_text)} | {teaser} | {esc(c.reason)} |")
    warns = list(a.rule_warnings or []) + [f"#{i} {w}" for i, c in enumerate(a.clips, 1) for w in c.warnings]
    if warns:
        md += ["", "## ⚠️ 規約・表現の注意"] + [f"- {w}" for w in warns]
    if prof:
        md += ["", f"## {prof.name}のルール（{prof.guideline_url}）"] + [f"- {n}" for n in prof.notes]
    md += ["", "## 投稿前チェックリスト",
           "- [ ] 配信者の切り抜きガイドライン（許可・収益化・クレジット表記）を確認した",
           "- [ ] 概要欄に元動画URLと配信者名を記載した",
           "- [ ] BGM・ゲーム映像などの第三者の権利物を確認した",
           "- [ ] 内輪ネタ・個人情報・誤解を招く切り取りになっていないか確認した"]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
