"""ASS 字幕の生成（折返し・強調語の色付け・タイトル帯）。"""
from __future__ import annotations

import re

from .config import Config
from .models import Segment

_BREAK_AFTER = set("、。,.!！?？ 　」』）)ー…")


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")


def wrap(text: str, width: int) -> list[str]:
    """width 文字ごとに折り返す。句読点・空白の直後で切れるなら優先する。"""
    text = text.strip()
    lines = []
    while len(text) > width:
        cut = width
        for i in range(width, max(width // 2, 1) - 1, -1):
            if text[i - 1] in _BREAK_AFTER:
                cut = i
                break
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return lines


def chunks(seg: Segment, max_chars: int) -> list[tuple[float, float, str]]:
    """1発話を「画面に出す単位」(最大 max_chars 文字)に分割し、時刻を割り当てる。"""
    if len(seg.text) <= max_chars:
        return [(seg.start, seg.end, seg.text)]

    if seg.words:  # 単語タイムスタンプがあれば正確に割る
        out, cur, cs = [], "", seg.words[0].start
        for w in seg.words:
            if cur and len(cur) + len(w.text) > max_chars:
                out.append((cs, w.start, cur.strip()))
                cur, cs = "", w.start
            cur += w.text
        if cur.strip():
            out.append((cs, seg.words[-1].end, cur.strip()))
        return out

    # 単語時刻がなければ文字数比で割る
    pieces = wrap(seg.text, max_chars)
    total = sum(len(p) for p in pieces) or 1
    out, t = [], seg.start
    for p in pieces:
        d = seg.duration * len(p) / total
        out.append((t, t + d, p))
        t += d
    return out


def highlight(line: str, keywords: list[str]) -> str:
    """強調語を黄色に（ASS の色は &HBBGGRR&）。"""
    line = escape(line)
    for kw in sorted(set(keywords), key=len, reverse=True):
        k = escape(kw)
        if k:
            line = re.sub(re.escape(k), lambda m: r"{\c&H00F0FF&}" + m.group(0) + r"{\r}", line)
    return line


def build_ass(events: list[tuple[float, float, str]], title: str, total: float,
              keywords: list[str], cfg: Config, video_h_ratio: float = 9 / 16) -> str:
    W, H = (cfg.width, cfg.height) if cfg.vertical else (1920, 1080)
    fs = cfg.font_size if cfg.vertical else 64
    if cfg.vertical:
        # 中央に置いた元映像のすぐ下に字幕、上の余白にタイトル帯
        fg_h = int(W * video_h_ratio)
        video_bottom = (H + fg_h) // 2
        sub_margin = max(80, H - video_bottom - int(fs * 2.6))
        title_margin = max(80, (H - fg_h) // 2 - int(fs * 2.2))
    else:
        sub_margin, title_margin = 60, 40

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{cfg.font},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{max(4, fs // 12)},2,2,40,40,{sub_margin},1
Style: Title,{cfg.font},{int(fs * 1.05)},&H00000000,&H000000FF,&H0000F0FF,&H0000F0FF,-1,0,0,0,100,100,0,0,3,14,0,8,60,60,{title_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        t = r"\N".join(escape(x) for x in wrap(title, 13)[:2])
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{t}")
    for s, e, text in events:
        if e - s < 0.05:
            continue
        body = r"\N".join(highlight(x, keywords) for x in wrap(text, cfg.max_line_chars)[:2])
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Sub,,0,0,0,,{body}")
    return header + "\n".join(lines) + "\n"
