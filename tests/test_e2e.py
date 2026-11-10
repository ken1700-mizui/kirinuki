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
