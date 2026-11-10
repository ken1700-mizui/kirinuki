"""ASS 字幕の生成（折返し・強調語の色付け・タイトル帯）。"""
from __future__ import annotations

import math
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


def _cls(ch: str) -> str:
    if re.match(r"[一-龥々〆ヵヶ]", ch):
        return "K"  # 漢字
    if re.match(r"[ぁ-ゖ]", ch):
        return "H"  # ひらがな
    if re.match(r"[ァ-ヺー]", ch):
        return "T"  # カタカナ
    if re.match(r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]", ch):
        return "L"
    return "P"


_NO_START = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮー、。,.!！?？）)」』…ゝゞ々")
_NO_END = set("（(「『")
# 前の文字種 → 次の文字種 の改行コスト（小さいほど自然な切れ目）
_PAIR_COST = {
    ("H", "K"): 1, ("H", "T"): 1, ("H", "L"): 1, ("K", "T"): 2, ("T", "K"): 2,
    ("H", "H"): 4, ("K", "H"): 5, ("T", "H"): 4, ("L", "H"): 2, ("L", "K"): 2, ("K", "L"): 2, ("T", "L"): 2,
    ("K", "K"): 8, ("T", "T"): 9, ("L", "L"): 12,
}


def _break_cost(text: str, i: int) -> float:
    """text[i-1] と text[i] の間で改行するコスト。"""
    a, b = text[i - 1], text[i]
    if b in _NO_START or a in _NO_END:
        return 50.0
    if a in _BREAK_AFTER:
        return 0.0
    if b in " 　":
        return 0.5
    if a in "にをはがのでともへやかな" and _cls(b) in ("K", "T", "L"):
        return 0.5  # 助詞の後ろ → 次の語の頭
    return float(_PAIR_COST.get((_cls(a), _cls(b)), 3))


def wrap(text: str, width: int) -> list[str]:
    """日本語向けの折り返し。

    - 1行は width 文字以内
    - 句読点・助詞の後ろなど自然な位置で切る（単語の途中・禁則を避ける）
    - 各行の長さをなるべく揃える（テロップは「下が長い/上下同じ」が読みやすい）
    動的計画法で「切れ目コスト + 行長のばらつき」の合計を最小化する。
    """
    text = re.sub(r"\s+", " ", text.strip())
    n = len(text)
    if n <= width:
        return [text] if text else []
    k = math.ceil(n / width)
    target = n / k
    INF = float("inf")
    # best[j] = 先頭 j 文字を行に分けたときの最小コスト, prev[j] = 直前の切れ目
    best = [INF] * (n + 1)
    prev = [0] * (n + 1)
    lines_used = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - width), j):
            if best[i] == INF:
                continue
            seg_len = j - i
            cost = best[i] + ((seg_len - target) / max(target, 1)) ** 2 * 6
            if j < n:
                cost += _break_cost(text, j) + 2.0  # 行数が増えるほど少し不利
            if cost < best[j]:
                best[j], prev[j], lines_used[j] = cost, i, lines_used[i] + 1
    out, j = [], n
    while j > 0:
        i = prev[j]
        out.append(text[i:j].strip())
        j = i
    return [x for x in reversed(out) if x]


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
              keywords: list[str], cfg: Config, video_h_ratio: float = 9 / 16,
              hook_text: str = "", hook_dur: float = 2.2) -> str:
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
Style: Hook,{cfg.font},{int(fs * 1.6)},&H00FFFFFF,&H000000FF,&H002020E0,&H80000000,-1,0,0,0,100,100,0,0,1,{max(6, fs // 8)},3,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        t = r"\N".join(escape(x) for x in wrap(title, 13))
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{t}")
    if hook_text:
        # 冒頭の一言: 画面中央に大きく、ポップイン → フェードアウト
        h = r"\N".join(escape(x) for x in wrap(hook_text, 9))
        fx = r"{\fad(80,250)\t(0,160,\fscx112\fscy112)\t(160,300,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 2,{ass_time(0)},{ass_time(min(hook_dur, total))},Hook,,0,0,0,,{fx}{h}")
    for s, e, text in events:
        if e - s < 0.05:
            continue
        body = r"\N".join(highlight(x, keywords) for x in wrap(text, cfg.max_line_chars))
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Sub,,0,0,0,,{body}")
    return header + "\n".join(lines) + "\n"
