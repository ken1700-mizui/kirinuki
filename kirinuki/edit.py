"""編集: ジャンプカット → (ティーザー) → 縦型化 → 字幕焼き込み → ラウドネス正規化。

ffmpeg の filter_complex 1本で完結させる。
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .config import Config
from .models import Clip, Segment
from .subtitles import build_ass, chunks

Range = tuple[float, float]


def keep_ranges(start: float, end: float, silences: list[Range], cfg: Config) -> list[Range]:
    """[start, end] から長い沈黙を詰めた「残す区間」のリスト。"""
    pad = cfg.jump_cut_keep / 2
    cuts = []
    for s, e in silences:
        s, e = max(s, start), min(e, end)
        if e - s >= cfg.jump_cut_min:
            cuts.append((s + pad, e - pad))
    out, t = [], start
    for s, e in sorted(cuts):
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if end > t:
        out.append((t, end))
    return [(a, b) for a, b in out if b - a >= 0.05]


def plan_teaser(clip: Clip, segs: list[Segment], cfg: Config) -> Range | None:
    """フックが弱いとき、ピークの発話を冒頭に最大 teaser_max 秒チラ見せする。"""
    if cfg.teaser == "off" or not segs:
        return None
    if clip.peak - clip.start < 5.0:  # すでに冒頭が山場
        return None
    hook = clip.features.get("hook", 0.0)
    peak = clip.features.get("peak", 0.0)
    if cfg.teaser == "auto" and not (peak > 0 and hook < peak * 0.6):
        return None
    seg = next((s for s in segs if s.start <= clip.peak + 0.5 <= s.end + 0.5 and s.end > clip.start), None)
    if seg is None:
        return None
    a = max(seg.start - 0.05, clip.start)
    b = min(seg.end + 0.3, a + cfg.teaser_max, clip.end)
    return (a, b) if b - a >= 0.8 else None


def subtitle_events(ranges: list[Range], segs: list[Segment], cfg: Config) -> list[tuple[float, float, str]]:
    """元動画の時刻の字幕を、出力（ジャンプカット後）の時刻に写像する。"""
    pieces = [c for s in segs for c in chunks(s, cfg.max_line_chars * 2)]
    events, offset = [], 0.0
    for a, b in ranges:
        for s, e, text in pieces:
            if e <= a or s >= b:
                continue
            os_, oe = offset + max(s, a) - a, offset + min(e, b) - a
            # 区間をまたいで同じ字幕が続くなら連結する
            if events and events[-1][2] == text and os_ - events[-1][1] < 0.05:
                events[-1] = (events[-1][0], oe, text)
            else:
                events.append((os_, oe, text))
        offset += b - a
    return events


def _audio_concat(rel: list[Range]) -> list[str]:
    f = [f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[m{i}]" for i, (a, b) in enumerate(rel)]
    f.append("".join(f"[m{i}]" for i in range(len(rel))) + f"concat=n={len(rel)}:v=0:a=1[mc]")
    return f


def _seek_window(ranges: list[Range]) -> tuple[float, float]:
    seek = max(0.0, min(a for a, _ in ranges) - 1.0)
    return seek, max(b for _, b in ranges) + 0.5 - seek


def measure_loudness(src: str, ranges: list[Range], cfg: Config) -> dict | None:
    """loudnorm 1パス目: 実際に使う区間だけの音量を測る（音声のみなので速い）。"""
    seek, span = _seek_window(ranges)
    rel = [(a - seek, b - seek) for a, b in ranges]
    f = _audio_concat(rel) + [f"[mc]loudnorm=I={cfg.loudness_i}:TP=-1.5:LRA=11:print_format=json[out]"]
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{seek:.3f}", "-t", f"{span:.3f}",
                        "-i", src, "-filter_complex", ";".join(f), "-map", "[out]", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr)
    if r.returncode != 0 or not m:
        return None
    d = json.loads(m.group(0))
    try:
        if float(d["input_i"]) < -70:  # ほぼ無音: 補正しない
            return None
    except ValueError:
        return None
    return d


def loudnorm_filter(cfg: Config, measured: dict | None) -> str:
    base = f"loudnorm=I={cfg.loudness_i}:TP=-1.5:LRA=11"
    if not measured:
        return base
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


def _frame_filter(cfg: Config, ass_name: str, src_label: str = "[0:v]", out_label: str = "[vout]") -> str:
    W, H = cfg.width, cfg.height
    if cfg.vertical:
        return (
            f"{src_label}fps={cfg.fps},split[bg0][fg0];"
            f"[bg0]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
            f"boxblur=8:2,scale={W}:{H},eq=brightness=-0.12[bg];"
            f"[fg0]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,ass={ass_name}{out_label}"
        )
    return (f"{src_label}fps={cfg.fps},scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,ass={ass_name}{out_label}")


def build_ffmpeg_cmd(src: str, out: str, ass_name: str, ranges: list[Range], has_audio: bool,
                     src_w: int, src_h: int, cfg: Config, measured: dict | None = None) -> list[str]:
    seek, span = _seek_window(ranges)
    rel = [(a - seek, b - seek) for a, b in ranges]
    total = sum(b - a for a, b in ranges)

    f = []
    vlabels, alabels = [], []
    for i, (a, b) in enumerate(rel):
        f.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        vlabels.append(f"[v{i}]")
        if has_audio:
            f.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
            alabels.append(f"[a{i}]")
    n = len(rel)
    if has_audio:
        f.append("".join(v + a for v, a in zip(vlabels, alabels)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    else:
        f.append("".join(vlabels) + f"concat=n={n}:v=1:a=0[vc]")

    f.append(_frame_filter(cfg, ass_name, "[vc]"))
    if has_audio:
        f.append(f"[ac]{loudnorm_filter(cfg, measured)},aresample=48000[aout]")

    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{seek:.3f}", "-t", f"{span:.3f}",
           "-i", src]
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-filter_complex", ";".join(f), "-map", "[vout]"]
    cmd += ["-map", "[aout]"] if has_audio else ["-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
    return cmd


def render(src: str | Path, clip: Clip, segs: list[Segment], silences: list[Range], info: dict,
           out_dir: str | Path, name: str, cfg: Config) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(src).resolve())

    main = keep_ranges(clip.start, clip.end, silences, cfg)
    if not main:
        main = [(clip.start, clip.end)]
    ranges = list(main)
    if clip.teaser:
        ranges = [clip.teaser] + ranges

    window_segs = [s for s in segs if s.end > clip.start and s.start < clip.end]
    events = subtitle_events(ranges, window_segs, cfg)
    total = sum(b - a for a, b in ranges)
    ratio = (info["height"] / info["width"]) if info.get("width") else 9 / 16
    ass = build_ass(events, clip.title, total, clip.keywords, cfg, video_h_ratio=min(ratio, 16 / 9),
                    hook_text=clip.hook_text)
    ass_name = f"{name}.ass"
    (out_dir / ass_name).write_text(ass, encoding="utf-8")

    out = out_dir / f"{name}.mp4"
    measured = measure_loudness(src, ranges, cfg) if info["has_audio"] else None
    cmd = build_ffmpeg_cmd(src, out.name, ass_name, ranges, info["has_audio"],
                           info.get("width", 0), info.get("height", 0), cfg, measured)
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗 ({name}):\n{r.stderr[-2000:]}")
    return out


def render_thumbnail(src: str | Path, clip: Clip, info: dict, out_dir: str | Path, name: str, cfg: Config) -> Path:
    """山場のフレーム＋大きなタイトルでサムネイルを作る。"""
    out_dir = Path(out_dir)
    W, H = (cfg.width, cfg.height) if cfg.vertical else (1920, 1080)
    fs = int((cfg.font_size if cfg.vertical else 64) * 1.7)
    title = clip.title or clip.hook_text
    from .subtitles import escape, wrap
    text = r"\N".join(escape(x) for x in wrap(title, 8 if cfg.vertical else 12))
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Thumb,{cfg.font},{fs},&H0000F0FF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{fs // 9},6,2,40,40,{int(H * 0.12)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Thumb,,0,0,0,,{text}
"""
    ass_name = f"{name}_thumb.ass"
    (out_dir / ass_name).write_text(ass, encoding="utf-8")
    out = out_dir / f"{name}.jpg"
    t = max(0.0, min(clip.peak, (info.get("duration") or clip.peak + 1) - 0.1))
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(Path(src).resolve()),
           "-filter_complex", _frame_filter(cfg, ass_name), "-map", "[vout]", "-frames:v", "1", "-q:v", "3", out.name]
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"サムネイル生成失敗 ({name}):\n{r.stderr[-1500:]}")
    return out
