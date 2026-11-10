"""Claude による候補の再ランク・境界調整・タイトル/フック生成。

LLMは「品質を上げる層」であり、失敗してもヒューリスティックの結果で処理を続ける。
"""
from __future__ import annotations

import json
import logging
import os

from .config import Config
from .highlight import clip_bounds
from .models import Clip, Segment

log = logging.getLogger(__name__)

RUBRIC = ("hook", "standalone", "payoff", "emotion", "shareability")
CONTEXT_BEFORE = 4
CONTEXT_AFTER = 3

SYSTEM_PROMPT = """あなたは日本トップクラスの切り抜き動画編集者です。
長時間配信の文字起こしから、ショート動画（15〜60秒）として最も伸びる区間を見極めます。

良い切り抜きの条件:
1. hook: 最初の3秒で視聴者を掴む（前置き・自己紹介から始まらない）
2. standalone: 元配信を見ていない人でも文脈が分かる
3. payoff: オチ・感情のピークがあり、その直後で終わる
4. emotion: 笑い・驚き・感動などの感情の強さ
5. shareability: 誰かに送りたくなる・コメントしたくなる

各候補には数値シグナル（音量の跳ね・笑い語・発話密度から計算した盛り上がり度）と、
前後の文脈を含む文字起こし（[id] 開始秒 テキスト）が付きます。
数値は「どう言ったか」、テキストは「何を言ったか」の手がかりです。両方を使ってください。

各候補について:
- 5観点を 0〜10 の整数で厳しめに採点する（平凡なら5、本当に強いものだけ8以上）
- 単体で成立し、オチで終わるように start_segment_id / end_segment_id を選び直す
  （提示された id の範囲内から選ぶ。尺は15〜60秒に収める）
- title: 何が起きたか＋感情が伝わる18文字前後の日本語タイトル（ネタバレしすぎない）
- hook_text: 冒頭に表示する短い一言（12文字以内）
- keywords: 字幕で色を付けて強調すべき語（文字起こし中に実在する語のみ、最大4個）
- reason: 選定理由を一文で

内輪ネタ・誹謗中傷・個人情報・センシティブな話題を含む候補は shareability を低くしてください。"""


def _schema() -> dict:
    item = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "integer"},
            **{k: {"type": "integer"} for k in RUBRIC},
            "start_segment_id": {"type": "integer"},
            "end_segment_id": {"type": "integer"},
            "title": {"type": "string"},
            "hook_text": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["candidate_id", *RUBRIC, "start_segment_id", "end_segment_id",
                     "title", "hook_text", "keywords", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"clips": {"type": "array", "items": item}},
        "required": ["clips"],
        "additionalProperties": False,
    }


def build_prompt(cands: list[Clip], segs: list[Segment]) -> str:
    parts = []
    for ci, c in enumerate(cands):
        i0, i1 = c.seg_range
        lo, hi = max(0, i0 - CONTEXT_BEFORE), min(len(segs) - 1, i1 + CONTEXT_AFTER)
        lines = []
        for j in range(lo, hi + 1):
            mark = "  " if i0 <= j <= i1 else "··"  # ·· は現在の範囲外（文脈）
            lines.append(f"{mark}[{j}] {segs[j].start:7.1f}s {segs[j].text}")
        f = c.features
        parts.append(
            f"## 候補 {ci}\n"
            f"現在の範囲: id {i0}〜{i1}（{c.start:.1f}s〜{c.end:.1f}s, {c.duration:.0f}秒）\n"
            f"盛り上がりピーク: {c.peak:.0f}s / peak={f.get('peak')} hook={f.get('hook')} "
            f"mean={f.get('mean')} payoff={f.get('payoff')}\n"
            + "\n".join(lines)
        )
    return ("以下の切り抜き候補をすべて評価してください。"
            "「··」の行は文脈で、範囲を広げる場合に使えます。\n\n" + "\n\n".join(parts))


def available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_PROFILE"))


def call_claude(prompt: str, cfg: Config, client=None) -> dict:
    import anthropic

    client = client or anthropic.Anthropic()
    with client.beta.messages.stream(
        model=cfg.model,
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": _schema()}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude が応答を辞退しました（refusal）")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("出力が max_tokens で途切れました")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return json.loads(text)


def apply_llm(result: dict, cands: list[Clip], segs: list[Segment], duration: float, cfg: Config) -> list[Clip]:
    """LLM 出力を検証して候補に反映し、総合スコアで並べ替える。"""
    max_h = max((c.score for c in cands), default=1.0) or 1.0
    by_id = {}
    for r in result.get("clips", []):
        cid = r.get("candidate_id")
        if isinstance(cid, int) and 0 <= cid < len(cands):
            by_id[cid] = r

    for ci, c in enumerate(cands):
        r = by_id.get(ci)
        if not r:
            continue
        scores = {k: int(min(10, max(0, r.get(k, 0)))) for k in RUBRIC}
        c.llm_scores = scores

        i0, i1 = c.seg_range
        lo, hi = max(0, i0 - CONTEXT_BEFORE), min(len(segs) - 1, i1 + CONTEXT_AFTER)
        n0, n1 = r.get("start_segment_id"), r.get("end_segment_id")
        if isinstance(n0, int) and isinstance(n1, int) and lo <= n0 <= n1 <= hi:
            s, e = clip_bounds(segs, n0, n1, cfg, duration)
            if cfg.min_duration * 0.8 <= e - s <= cfg.max_duration * 1.1:
                c.start, c.end, c.seg_range = s, e, (n0, n1)

        c.title = str(r.get("title", "")).strip()[:40]
        c.hook_text = str(r.get("hook_text", "")).strip()[:20]
        window_text = "".join(segs[j].text for j in range(c.seg_range[0], c.seg_range[1] + 1))
        c.keywords = [k for k in r.get("keywords", []) if isinstance(k, str) and k and k in window_text][:4]
        c.reason = str(r.get("reason", "")).strip()

        llm_norm = sum(scores.values()) / (10 * len(RUBRIC))
        c.features["heuristic"] = round(c.score, 3)
        c.score = (1 - cfg.llm_weight) * (c.score / max_h) + cfg.llm_weight * llm_norm
    return sorted(cands, key=lambda c: -c.score)


def rerank(cands: list[Clip], segs: list[Segment], duration: float, cfg: Config, client=None) -> list[Clip]:
    top = sorted(cands, key=lambda c: -c.score)[: cfg.llm_candidates]
    if not top or not segs:
        return cands
    try:
        result = call_claude(build_prompt(top, segs), cfg, client=client)
    except Exception as e:  # LLM は任意層。落ちても続行する
        log.warning("LLM 再ランクをスキップ: %s", e)
        return cands
    return apply_llm(result, top, segs, duration, cfg)
