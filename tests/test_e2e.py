"""合成配信での結合テスト（ffmpeg 必須）。"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import HIGHLIGHT_END, HIGHLIGHT_START, LONG_SILENCE, make_stream  # noqa: E402

from kirinuki.audio import probe  # noqa: E402
from kirinuki.config import Config  # noqa: E402
from kirinuki.pipeline import analyze, run  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    return make_stream(tmp_path_factory.mktemp("fx"))


def test_detects_planted_highlight(stream):
    video, srt, _ = stream
    a = analyze(video, Config(use_llm=False, num_clips=3), transcript=srt)
    top = a.clips[0]
    inter = min(top.end, HIGHLIGHT_END) - max(top.start, HIGHLIGHT_START)
    assert inter / (HIGHLIGHT_END - HIGHLIGHT_START) > 0.7, top
    # 数合わせの弱いクリップは出さない
    assert all(c.score >= top.score * 0.5 for c in a.clips)


def test_long_silence_detected(stream):
    video, srt, _ = stream
    a = analyze(video, Config(use_llm=False), transcript=srt)
    assert any(s <= LONG_SILENCE[0] + 0.2 and e >= LONG_SILENCE[1] - 0.2 for s, e in a.timeline.silences)


def test_render_vertical(stream, tmp_path):
    video, srt, _ = stream
    a = run(video, tmp_path, Config(use_llm=False, num_clips=1), transcript=srt)
    out = tmp_path / "clip_01.mp4"
    info = probe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    c = a.clips[0]
    expected = c.duration + ((c.teaser[1] - c.teaser[0]) if c.teaser else 0)
    assert abs(info["duration"] - expected) < 1.0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["clips"][0]["file"].endswith("clip_01.mp4")
    assert (tmp_path / "clip_01.jpg").stat().st_size > 10_000
    assert "ガイドライン" in (tmp_path / "report.md").read_text()
    # ラウドネス -14 LUFS ±1.5
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(out), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    lufs = float(r.stderr.split("Summary:")[1].split("I:")[1].split("LUFS")[0])
    assert abs(lufs + 14) < 1.5


@pytest.fixture(scope="module")
def tricky(tmp_path_factory):
    from fixtures import make_chat
    d = tmp_path_factory.mktemp("tricky")
    video, srt, _ = make_stream(d, decoy=True)
    return video, srt, make_chat(d / "x.live_chat.json")


def _hits(clip, span, ratio=0.6):
    inter = min(clip.end, span[1]) - max(clip.start, span[0])
    return inter / (span[1] - span[0]) >= ratio


def test_ignores_loud_bgm_decoy(tricky):
    from fixtures import DECOY
    video, srt, _ = tricky
    a = analyze(video, Config(use_llm=False, num_clips=5), transcript=srt)
    assert _hits(a.clips[0], (HIGHLIGHT_START, HIGHLIGHT_END))
    assert not any(_hits(c, DECOY, 0.5) for c in a.clips), [(c.start, c.end) for c in a.clips]


def test_chat_finds_visual_only_moment(tricky):
    from fixtures import CHAT_ONLY, DECOY
    video, srt, chat = tricky
    a = analyze(video, Config(use_llm=False, num_clips=5), transcript=srt, chat=chat)
    assert _hits(a.clips[0], (HIGHLIGHT_START, HIGHLIGHT_END))
    assert any(_hits(c, CHAT_ONLY, 0.8) for c in a.clips[:2])
    assert not any(_hits(c, DECOY, 0.5) for c in a.clips)
    # 各クリップの主役は自分のピーク（他の山を取り込まない）
    assert all(c.start <= c.peak <= c.end for c in a.clips)


def test_truth_locate_and_review(tricky, tmp_path):
    from kirinuki.evaluate import build_truth, score
    from kirinuki.review import review
    video, srt, chat = tricky
    # 「人間の切り抜き」を模擬: 山場をジャンプカット＋音量変更＋ノイズ
    ref = tmp_path / "ref.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-filter_complex",
                    "[0:v]trim=118:126,setpts=PTS-STARTPTS[v1];[0:a]atrim=118:126,asetpts=PTS-STARTPTS[a1];"
                    "[0:v]trim=127:136,setpts=PTS-STARTPTS[v2];[0:a]atrim=127:136,asetpts=PTS-STARTPTS[a2];"
                    "[v1][a1][v2][a2]concat=n=2:v=1:a=1[v][a0];[a0]volume=-5dB[a]",
                    "-map", "[v]", "-map", "[a]", str(ref)], check=True)
    truth = build_truth(video, [ref])
    assert truth[0]["found"] and abs(truth[0]["start"] - 118) < 1 and abs(truth[0]["end"] - 136) < 1.5

    a = run(video, tmp_path / "out", Config(use_llm=False, num_clips=2), transcript=srt, chat=chat,
            profile="noripro")
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert score(rep["clips"], truth)["recall@1"] == 1.0
    assert "元配信" in (tmp_path / "out" / "clip_01.txt").read_text()

    md = review(tmp_path / "out" / "clip_01.mp4", [ref], tmp_path / "rv")
    assert md.exists() and (tmp_path / "rv" / "clip_01_sheet.jpg").exists()
    assert "冒頭3秒の勢い" in md.read_text()
