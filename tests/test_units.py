import json
from types import SimpleNamespace

import numpy as np
import pytest

from kirinuki import llm
from kirinuki.audio import AudioAnalysis, local_jump
from kirinuki.chat import chat_signal, load_chat
from kirinuki.config import Config
from kirinuki.edit import keep_ranges, plan_teaser, subtitle_events
from kirinuki.highlight import Timeline, generate_candidates, overlap, select
from kirinuki.models import Clip, Segment, Word
from kirinuki.subtitles import ass_time, build_ass, chunks, highlight, wrap
from kirinuki.text import reaction_score, starts_with_connective
from kirinuki.transcribe import load_srt, normalize


# ---------- text ----------
@pytest.mark.parametrize("text", ["草www", "やばいやばい！！", "ははははは", "うそでしょ！？", "lol no way"])
def test_reaction_positive(text):
    assert reaction_score(text) > 0.3


@pytest.mark.parametrize("text", ["今日は雑談します", "コメント読んでいきますね", "window を開く"])
def test_reaction_neutral(text):
    assert reaction_score(text) < 0.15


def test_connective():
    assert starts_with_connective("で、それがさ")
    assert not starts_with_connective("今日はね")


# ---------- transcribe ----------
def test_load_srt(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:02,500\nこんにちは\n\n2\n00:01:00.000 --> 00:01:01.000\n<i>テスト</i>\n",
                 encoding="utf-8")
    segs = load_srt(p)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.5, "こんにちは"), (60.0, 61.0, "テスト")]


def test_normalize_overlap():
    segs = normalize([Segment(0, 3, "a"), Segment(2, 4, "b"), Segment(5, 5, "empty")])
    assert len(segs) == 2 and segs[0].end <= segs[1].start


# ---------- audio ----------
def test_silences_and_jump():
    db = np.array([-20.0] * 20 + [-70.0] * 20 + [-20.0] * 20)
    a = AudioAnalysis(0.05, db)
    sil = a.silences(min_dur=0.3)
    assert len(sil) == 1 and sil[0] == pytest.approx((1.0, 2.0))
    x = np.array([-30.0] * 100)
    x[50] = -10
    j = local_jump(x)
    assert j.argmax() == 50 and j[50] == 1.0 and j[10] == 0


# ---------- edit ----------
def test_keep_ranges_jump_cut():
    cfg = Config()
    r = keep_ranges(10, 20, [(12, 14), (15, 15.3)], cfg)  # 2秒の沈黙は詰め、0.3秒は残す
    assert r[0] == (10, pytest.approx(12.075)) and r[1][0] == pytest.approx(13.925) and r[-1][1] == 20
    assert sum(b - a for a, b in r) == pytest.approx(10 - 2 + 0.15)


def test_subtitle_mapping_after_cut():
    cfg = Config()
    segs = [Segment(10, 12, "前半"), Segment(14, 16, "後半")]
    ev = subtitle_events([(10, 12.1), (13.9, 16)], segs, cfg)
    assert ev[0][2] == "前半" and ev[1][2] == "後半"
    assert ev[1][0] == pytest.approx(2.1 + 0.1)  # 出力側の時刻に写像されている


def test_teaser_auto():
    cfg = Config()
    segs = [Segment(0, 3, "前フリ"), Segment(20, 22, "草www")]
    c = Clip(0, 25, peak=20.5, features={"hook": 0.1, "peak": 0.9})
    assert plan_teaser(c, segs, cfg) == (pytest.approx(19.95), pytest.approx(22.3))
    c.features["hook"] = 0.8
    assert plan_teaser(c, segs, cfg) is None


# ---------- subtitles ----------
def test_wrap_and_chunks():
    assert all(len(x) <= 14 for x in wrap("これはとても長い字幕のテストです、句読点で切れるかな", 14))
    seg = Segment(0, 6, "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも")
    cs = chunks(seg, 14)
    assert len(cs) == 3 and cs[0][0] == 0 and cs[-1][1] == pytest.approx(6)
    seg_w = Segment(0, 4, "x" * 30, [Word(0, 1, "x" * 10), Word(1, 2, "x" * 10), Word(2, 4, "x" * 10)])
    assert [c[0] for c in chunks(seg_w, 20)] == [0, 2]


def test_ass_output():
    assert ass_time(3723.456) == "1:02:03.46"
    assert "{\\c&H00F0FF&}草{\\r}" in highlight("草生える", ["草"])
    assert "｛" in highlight("{悪意}", [])
    ass = build_ass([(0, 1, "テスト")], "タイトル", 10, [], Config())
    assert "Style: Title" in ass and "Dialogue: 1,0:00:00.00,0:00:10.00,Title" in ass


# ---------- highlight ----------
def _tl(excite):
    z = np.zeros(len(excite))
    return Timeline(z, z, z, np.asarray(excite, float), [])


def test_candidates_cover_peak_and_respect_duration():
    cfg = Config()
    segs = [Segment(i * 4.0, i * 4.0 + 3.5, f"文{i}") for i in range(30)]
    ex = np.full(120, 0.05)
    ex[70:74] = 0.9
    cands = generate_candidates(segs, _tl(ex), 120.0, cfg)
    best = max(cands, key=lambda c: c.score)
    assert best.start <= 70 <= best.end
    assert cfg.min_duration <= best.duration <= cfg.max_duration
    assert best.end - 74 < cfg.max_peak_to_end + 4  # 山場の後で終わる


def test_select_nms_and_floor():
    a, b, c = Clip(0, 30, 10, score=1.0), Clip(5, 35, 20, score=0.9), Clip(100, 130, 110, score=0.3)
    assert overlap(a, b) > 0.5
    assert select([a, b, c], 5, 0.3) == [a, c]
    assert select([a, b, c], 5, 0.3, min_relative=0.5) == [a]


# ---------- chat ----------
def test_chat_ytdlp_and_signal(tmp_path):
    lines = []
    for t in range(0, 100):
        n = 12 if 50 <= t < 53 else 1
        for _ in range(n):
            txt = "草" if n > 1 else "こんばんは"
            lines.append(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(t * 1000), "actions": [
                {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": txt}]}}}}}]}}))
    p = tmp_path / "x.live_chat.json"
    p.write_text("\n".join(lines), encoding="utf-8")
    msgs = load_chat(p)
    assert len(msgs) == 97 + 36
    sig = chat_signal(msgs, 100, delay=5)
    assert 44 <= int(sig.argmax()) <= 49  # 遅延ぶん前にずれる
    assert sig[10] == 0


def test_chat_generic_json_and_csv(tmp_path):
    (tmp_path / "a.json").write_text('[{"time": 1.5, "text": "w"}]')
    (tmp_path / "b.csv").write_text("time,text\n2,草\n")
    assert load_chat(tmp_path / "a.json") == [(1.5, "w")]
    assert load_chat(tmp_path / "b.csv") == [(2.0, "草")]


# ---------- llm ----------
class FakeClient:
    def __init__(self, payload=None, stop="end_turn", raise_exc=None):
        self.payload, self.stop, self.raise_exc, self.kwargs = payload, stop, raise_exc, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.kwargs = kw
        if self.raise_exc:
            raise self.raise_exc
        msg = SimpleNamespace(stop_reason=self.stop,
                              content=[SimpleNamespace(type="text", text=json.dumps(self.payload, ensure_ascii=False))])
        cm = SimpleNamespace(get_final_message=lambda: msg)
        return SimpleNamespace(__enter__=lambda *a: cm, __exit__=lambda *a: False)


def _fake_stream_factory(fake):
    class Ctx:
        def __enter__(self_inner):
            msg = SimpleNamespace(stop_reason=fake.stop, content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=json.dumps(fake.payload, ensure_ascii=False))])
            return SimpleNamespace(get_final_message=lambda: msg)

        def __exit__(self_inner, *a):
            return False

    def stream(**kw):
        fake.kwargs = kw
        if fake.raise_exc:
            raise fake.raise_exc
        return Ctx()
    fake.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    return fake


def _llm_fixture():
    segs = [Segment(i * 5.0, i * 5.0 + 4.5, f"発話{i}" + ("草www" if i == 6 else "")) for i in range(20)]
    # rerank はスコア順に並べてから id を振るので、ここではスコア降順で用意する
    cands = [Clip(20, 40, 30, score=0.6, features={}, seg_range=(4, 7)),
             Clip(60, 80, 70, score=0.5, features={}, seg_range=(12, 15))]
    return segs, cands


def test_llm_rerank_applies_scores_and_bounds():
    segs, cands = _llm_fixture()
    payload = {"clips": [
        {"candidate_id": 0, "hook": 9, "standalone": 9, "payoff": 10, "emotion": 9, "shareability": 9,
         "start_segment_id": 3, "end_segment_id": 7, "title": "草が生える瞬間", "hook_text": "まさかの",
         "keywords": ["草www", "存在しない語"], "reason": "オチが強い"},
        {"candidate_id": 1, "hook": 2, "standalone": 3, "payoff": 2, "emotion": 2, "shareability": 2,
         "start_segment_id": 0, "end_segment_id": 19, "title": "微妙", "hook_text": "", "keywords": [],
         "reason": "弱い"},
    ]}
    fake = _fake_stream_factory(FakeClient(payload))
    out = llm.rerank(cands, segs, 100.0, Config(), client=fake)
    assert out[0].title == "草が生える瞬間"
    assert out[0].seg_range == (3, 7) and out[0].start == pytest.approx(15 - 0.15)
    assert out[0].keywords == ["草www"]          # 文字起こしに無い語は捨てる
    assert out[1].seg_range == (12, 15)          # 範囲外の境界提案は無視
    kw = fake.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "[6]" in kw["messages"][0]["content"]


def test_llm_failure_falls_back():
    segs, cands = _llm_fixture()
    fake = _fake_stream_factory(FakeClient(raise_exc=RuntimeError("network")))
    assert llm.rerank(cands, segs, 100.0, Config(), client=fake) == cands
    fake = _fake_stream_factory(FakeClient(payload={}, stop="refusal"))
    assert llm.rerank(cands, segs, 100.0, Config(), client=fake) == cands


# ---------- source / hook ----------
def test_ytdlp_cmd_and_outputs(tmp_path):
    from kirinuki.source import find_outputs, is_url, ytdlp_cmd
    assert is_url("https://youtu.be/x") and not is_url("a.mp4")
    cmd = ytdlp_cmd("https://youtu.be/x", tmp_path)
    assert "live_chat,ja,ja-orig" in cmd and "--write-auto-subs" in cmd
    assert "--cookies" in ytdlp_cmd("u", tmp_path, cookies="c.txt")
    for n in ("source.mp4", "source.live_chat.json", "source.ja.vtt"):
        (tmp_path / n).write_text("x")
    src = find_outputs(tmp_path)
    assert src.video.name == "source.mp4" and src.chat.name.endswith("live_chat.json")
    assert src.subtitles.name == "source.ja.vtt"


def test_hook_overlay_in_ass():
    ass = build_ass([], "タイトル", 10, [], Config(), hook_text="まさかの展開")
    assert "Style: Hook" in ass and ",Hook,," in ass and "まさかの展開" in ass
    assert "Hook" not in build_ass([], "t", 10, [], Config()).split("[Events]")[1]


@pytest.mark.parametrize("text,width,expected", [
    ("油断した瞬間に完全敗北する配信者", 8, ["油断した瞬間に", "完全敗北する", "配信者"]),
    ("ちょっと待ってって言ったじゃん！！", 14, ["ちょっと待ってって", "言ったじゃん！！"]),
    ("アイテムはまだ取っておこうかな", 14, ["アイテムはまだ", "取っておこうかな"]),
])
def test_wrap_japanese_quality(text, width, expected):
    assert wrap(text, width) == expected


def test_wrap_never_drops_text():
    for t in ["Hello world this is English text", "あ" * 40, "ーーーーーーーーーーーーーーーーーー", "草" * 29]:
        lines = wrap(t, 14)
        assert "".join(lines).replace(" ", "") == t.replace(" ", "")
        assert all(len(x) <= 14 for x in lines)


# ---------- profiles ----------
def test_profiles_rules():
    from kirinuki import profiles
    p = profiles.get("noripro")
    err, _ = profiles.check_source(p, {"channel": "犬山たまき / 佃煮のりお", "title": "雑談"})
    assert err
    err, warn = profiles.check_source(p, {"channel": "レグルシュ・ライオンハート", "title": "【歌枠】耐久"})
    assert not err and warn
    err, _ = profiles.check_source(profiles.get("hololive"), {"channel": "x", "title": "【メン限】雑談"})
    assert err
    assert profiles.check_title("【放送事故】ブチギレ") and not profiles.check_title("油断した瞬間に完全敗北")
    c = Clip(125, 150, 130, title="タイトル")
    d = profiles.make_description(profiles.get("hololive"), {"title": "元配信", "webpage_url": "https://youtu.be/x",
                                                              "channel": "ch"}, c)
    assert d.startswith("【元配信】元配信\nhttps://youtu.be/x") and "https://youtu.be/x?t=125s" in d


# ---------- locate / evaluate ----------
def test_consistent_chain_rejects_outliers():
    from kirinuki.locate import _consistent_chain
    hits = [(0, 100, .9), (1, 101, .9), (2, 50, .95), (3, 103, .9), (4, 110, .9)]  # 2秒目は誤一致、4秒目はジャンプカット
    chain = _consistent_chain(hits, max_jump=60)
    assert [h[1] for h in chain] == [100, 101, 103, 110]


def test_score_recall_and_missed():
    from kirinuki.evaluate import score
    truth = [{"start": 100, "end": 130, "views": 9000}, {"start": 500, "end": 520, "views": 1000}]
    pred = [{"start": 98, "end": 128}, {"start": 300, "end": 330}]
    r = score(pred, truth)
    assert r["recall@1"] == 0.5 and r["view_recall@1"] == 0.9 and r["precision"] == 0.5
    assert r["missed"][0]["start"] == 500


def test_load_json3_autosub(tmp_path):
    from kirinuki.transcribe import load_transcript
    data = {"events": [
        {"tStartMs": 1000, "dDurationMs": 1500, "segs": [{"utf8": "ちょっと"}, {"utf8": "待って", "tOffsetMs": 600}]},
        {"tStartMs": 2600, "dDurationMs": 1000, "segs": [{"utf8": "やばい"}]},
        {"tStartMs": 3000, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 9000, "dDurationMs": 1000, "segs": [{"utf8": "次の話"}]},
    ]}
    p = tmp_path / "a.ja.json3"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    segs = load_transcript(p)
    assert [s.text for s in segs] == ["ちょっと待ってやばい", "次の話"]
    assert segs[0].words[1].start == pytest.approx(1.6) and segs[0].start == 1.0
