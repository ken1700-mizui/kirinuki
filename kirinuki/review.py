"""出力動画のレビュー: コマ割り画像 + 数値指標 + 伸びている切り抜きとの差分。

人間（またはClaude）が「見て評価する」ための材料を作る。
- sheet.jpg: 冒頭3秒を0.5秒刻み + 全体を均等に12コマ（タイムスタンプ付き）
- 指標: 尺・カット頻度・沈黙・音量・冒頭の勢い・字幕密度
- 参照（伸びている切り抜き）の中央値と比べ、外れている項目に印を付ける
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from pathlib import Path

from .audio import analyze_audio, probe

FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

# 指標の表示名・単位・「どちらが良いか」の目安
METRICS = {
    "duration": ("尺", "秒"),
    "cuts_per_min": ("カット頻度", "回/分"),
    "speech_ratio": ("発話率", ""),
    "longest_silence": ("最長の沈黙", "秒"),
    "loudness": ("音量", "LUFS"),
    "hook_energy": ("冒頭3秒の勢い", "dB"),
    "sub_coverage": ("字幕の表示率", ""),
    "sub_cps": ("字幕の文字/秒", ""),
    "title_len": ("タイトル文字数", "字"),
}


TOLERANCE = {"duration": 5.0, "cuts_per_min": 3.0, "speech_ratio": 0.08, "longest_silence": 0.4,
             "loudness": 1.5, "hook_energy": 2.0, "sub_coverage": 0.1, "sub_cps": 1.5, "title_len": 5}


def _loudness(path: str) -> float | None:
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"Summary:.*?I:\s*(-?[\d.]+) LUFS", r.stderr, re.S)
    return float(m.group(1)) if m else None


def _scene_cuts(path: str, thresh: float = 0.3) -> int:
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-an", "-vf",
                        f"scale=160:-2,select='gt(scene,{thresh})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return len(re.findall(r"Parsed_showinfo.*?pts_time", r.stderr))


def _ass_stats(ass: Path, duration: float) -> dict:
    events = []
    for line in ass.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:") and ",Sub," in line:
            parts = line.split(",", 9)
            def t(x):
                h, m, s = x.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            text = re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", "")
            events.append((t(parts[1]), t(parts[2]), text))
    if not events or duration <= 0:
        return {"sub_coverage": 0.0, "sub_cps": 0.0}
    shown = sum(e - s for s, e, _ in events)
    chars = sum(len(x) for *_, x in events)
    return {"sub_coverage": round(min(shown / duration, 1.0), 2), "sub_cps": round(chars / max(shown, 1e-6), 1)}


def metrics(video: str | Path, ass: str | Path | None = None, title: str | None = None) -> dict:
    video = str(video)
    info = probe(video)
    dur = info["duration"]
    m: dict = {"file": Path(video).name, "duration": round(dur, 1),
               "aspect": f"{info['width']}x{info['height']}"}
    if info["has_audio"]:
        au = analyze_audio(video)
        sil = au.silences(0.3)
        m["speech_ratio"] = round(1 - sum(e - s for s, e in sil) / max(dur, 1e-6), 2)
        m["longest_silence"] = round(max((e - s for s, e in sil), default=0.0), 2)
        per = au.per_second(max(1, int(dur)))
        if len(per) >= 4:
            m["hook_energy"] = round(float(per[:3].mean() - per.mean()), 1)
        m["loudness"] = _loudness(video)
    if info["has_video"]:
        m["cuts_per_min"] = round(_scene_cuts(video) / max(dur / 60, 1e-6), 1)
    ass_path = Path(ass) if ass else Path(video).with_suffix(".ass")
    if ass_path.exists():
        m.update(_ass_stats(ass_path, dur))
    meta = Path(video).with_suffix(".info.json")
    if meta.exists():  # yt-dlp のメタデータ（参照動画の再生数・タイトル）
        d = json.loads(meta.read_text(encoding="utf-8"))
        title = title or d.get("title")
        m["views"] = d.get("view_count")
    if title:
        m["title"] = title
        m["title_len"] = len(title)
    return m


def contact_sheet(video: str | Path, out: str | Path) -> Path:
    """冒頭3秒(0.5秒刻み6コマ) + 全体12コマ のコマ割り画像。"""
    video, out = str(video), Path(out)
    dur = probe(video)["duration"]
    # 縮小してから時刻を描く（先に描くと潰れて読めない）
    stamp = (f"scale=240:-2,drawtext=fontfile={FONT}:text='%{{pts\\:hms}}':x=4:y=4:fontsize=18:"
             "fontcolor=white:box=1:boxcolor=black@0.7")
    hook = out.with_name(out.stem + "_hook.png")
    body = out.with_name(out.stem + "_body.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "3", "-i", video, "-vf",
                    f"fps=2,{stamp},tile=6x1", "-frames:v", "1", str(hook)], check=True)
    rate = 12 / max(dur, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf",
                    f"fps={rate:.5f},{stamp},tile=6x2", "-frames:v", "1", str(body)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(hook), "-i", str(body), "-filter_complex",
                    "[0][1]vstack", str(out)],
                   check=False, capture_output=True)
    if not out.exists():  # 幅が違う場合のフォールバック
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(hook), "-i", str(body), "-filter_complex",
                        "[0]scale=1440:-2[a];[1]scale=1440:-2[b];[a][b]vstack", str(out)], check=True)
    hook.unlink(missing_ok=True)
    body.unlink(missing_ok=True)
    return out


def compare(ours: dict, refs: list[dict]) -> list[dict]:
    """参照の中央値・範囲と比べ、外れている指標を返す。"""
    rows = []
    for key, (label, unit) in METRICS.items():
        vals = [r[key] for r in refs if isinstance(r.get(key), (int, float))]
        v = ours.get(key)
        if not vals or not isinstance(v, (int, float)):
            continue
        med, lo, hi = statistics.median(vals), min(vals), max(vals)
        # 参照が少ないと範囲が狭すぎるので、中央値の30%（最低でも指標ごとの許容幅）は誤差とみなす
        tol = max((hi - lo) * 0.5, abs(med) * 0.3, TOLERANCE.get(key, 0.0))
        flag = ""
        if v < lo - tol:
            flag = "低い"
        elif v > hi + tol:
            flag = "高い"
        rows.append({"key": key, "label": label, "unit": unit, "ours": v, "ref_median": round(med, 2),
                     "ref_range": [round(lo, 2), round(hi, 2)], "flag": flag})
    return rows


def review(video: str | Path, refs: list[str | Path], out_dir: str | Path, title: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ours = metrics(video, title=title)
    sheet = contact_sheet(video, out_dir / f"{Path(video).stem}_sheet.jpg")
    ref_m = []
    for r in refs:
        rm = metrics(r)
        rm["sheet"] = str(contact_sheet(r, out_dir / f"ref_{Path(r).stem}_sheet.jpg"))
        ref_m.append(rm)
    rows = compare(ours, ref_m)

    md = [f"# レビュー: {Path(video).name}", "", f"![sheet]({sheet.name})", "",
          "## 指標", "", "| 指標 | 自分 | 参照の中央値 | 参照の範囲 | 判定 |", "|---|---|---|---|---|"]
    if rows:
        for r in rows:
            md.append(f"| {r['label']} | {r['ours']}{r['unit']} | {r['ref_median']} | "
                      f"{r['ref_range'][0]}〜{r['ref_range'][1]} | {('⚠️ ' + r['flag']) if r['flag'] else '✅'} |")
    else:
        for k, (label, unit) in METRICS.items():
            if k in ours:
                md.append(f"| {label} | {ours[k]}{unit} | - | - | - |")
    if ref_m:
        md += ["", "## 参照した切り抜き", "", "| ファイル | 再生数 | 尺 | タイトル |", "|---|---|---|---|"]
        for r in ref_m:
            md.append(f"| {r['file']} | {r.get('views', '-')} | {r.get('duration')} | {r.get('title', '-')} |")
    md += ["", "## 目視チェック（レビュアーが記入）", "",
           "- [ ] 冒頭1秒で何の動画か分かる", "- [ ] 冒頭3秒で続きを見たくなる",
           "- [ ] 文の途中から始まっていない", "- [ ] オチの後に余韻があり、ブツ切りでない",
           "- [ ] 字幕が読める速さ・位置・大きさ", "- [ ] 強調語の色付けが効いている",
           "- [ ] タイトルが内容を正しく表し、誇張していない", "- [ ] 無駄な間がない"]
    report = out_dir / f"{Path(video).stem}_review.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / f"{Path(video).stem}_review.json").write_text(
        json.dumps({"ours": ours, "refs": ref_m, "compare": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
