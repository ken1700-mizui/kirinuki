"""見どころ検出: 盛り上がり度の時系列 → 候補窓の生成 → 採点 → 重複除去。"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from .audio import AudioAnalysis, loudness_jump
from .config import Config
from .models import Clip, Segment
from .text import char_count, reaction_score, starts_with_connective


@dataclass
class Timeline:
    """1秒解像度のシグナル群。"""
    loud: np.ndarray
    react: np.ndarray
    density: np.ndarray
    excite: np.ndarray
    silences: list[tuple[float, float]]
    chat: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.excite)


def _smooth(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.copy()
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(np.pad(x, 1, mode="edge"), k, mode="valid")


def build_timeline(segs: list[Segment], audio: AudioAnalysis, duration: float, cfg: Config,
                   chat: np.ndarray | None = None) -> Timeline:
    n = max(1, int(math.ceil(duration)))
    loud = loudness_jump(audio.per_second(n)) if len(audio.db) else np.zeros(n)

    react = np.zeros(n)
    dens_raw = np.zeros(n)
    for s in segs:
        a, b = int(s.start), min(n, int(math.ceil(s.end)))
        if b <= a:
            continue
        r = reaction_score(s.text)
        react[a:b] = np.maximum(react[a:b], r)
        cps = char_count(s.text) / max(s.duration, 0.5)
        dens_raw[a:b] = np.maximum(dens_raw[a:b], cps)

    spoken = dens_raw[dens_raw > 0]
    density = np.zeros(n)
    if len(spoken) > 2:
        mu, sd = spoken.mean(), max(spoken.std(), 1e-6)
        density = np.where(dens_raw > 0, np.clip((dens_raw - mu) / sd, 0, 3) / 3, 0.0)

    chat_arr = np.zeros(n)
    has_chat = chat is not None and len(chat) > 0 and bool(np.any(chat))
    if has_chat:
        chat_arr[: min(n, len(chat))] = chat[:n]

    # 音量の裏取り: 「うるさいだけ」(BGM・効果音) を山場と誤認しないよう、
    # 近く(±3秒)に笑い語かチャットの反応があるときだけ音量を満点で数える。
    # （早口は話題によって常に高いことがあり、裏付けとしては弱いので使わない）
    if segs or has_chat:
        support = np.maximum(react, chat_arr)
        support = np.array([support[max(0, t - 3):t + 4].max() for t in range(n)])
        loud_eff = loud * (cfg.loud_gate_min + (1 - cfg.loud_gate_min) * np.clip(support * 2, 0, 1))
    else:
        loud_eff = loud  # 文字起こしもチャットも無いときは音量が唯一の手がかり

    # ノイジーOR: どれか1つの手がかりが強ければ山場になり、複数が重なるほど確信が増す
    miss = (1 - cfg.w_loud * loud_eff) * (1 - cfg.w_react * react) * (1 - cfg.w_density * density)
    if has_chat:
        miss = miss * (1 - cfg.w_chat * chat_arr)
    excite = _smooth(1 - miss)
    return Timeline(loud, react, density, excite, audio.silences(cfg.silence_min), chat_arr)


def find_peaks(excite: np.ndarray, radius: int = 5, k: int = 20) -> list[int]:
    """近傍 radius 秒内で最大、かつ上位の点をピークとする。"""
    if len(excite) == 0:
        return []
    floor = max(float(np.percentile(excite, 75)), 0.08)
    peaks = []
    for t in range(len(excite)):
        lo, hi = max(0, t - radius), min(len(excite), t + radius + 1)
        v = excite[t]
        if v >= floor and v == excite[lo:hi].max() and (not peaks or t - peaks[-1] > radius):
            peaks.append(t)
    peaks.sort(key=lambda t: -excite[t])
    return peaks[:k]


def _gap_before(segs: list[Segment], i: int) -> float:
    return segs[i].start - segs[i - 1].end if i > 0 else 5.0


def _seg_at(starts: list[float], segs: list[Segment], t: float) -> int:
    i = bisect.bisect_right(starts, t) - 1
    return max(0, min(i, len(segs) - 1))


def clip_bounds(segs: list[Segment], i0: int, i1: int, cfg: Config, duration: float) -> tuple[float, float]:
    """セグメント範囲 → 実際の開始/終了時刻（文頭リードイン・余韻つき、隣の発話を食わない）。"""
    start = segs[i0].start - cfg.lead_in
    if i0 > 0:
        start = max(start, segs[i0 - 1].end + 0.02)
    end = segs[i1].end + cfg.tail
    if i1 + 1 < len(segs):
        end = min(end, max(segs[i1].end + 0.1, segs[i1 + 1].start - 0.05))
    return max(0.0, start), min(duration, end)


def score_window(tl: Timeline, start: float, end: float, cfg: Config) -> tuple[float, dict]:
    a, b = int(start), max(int(start) + 1, int(math.ceil(end)))
    ex = tl.excite[a:min(b, tl.n)]
    if len(ex) == 0:
        return 0.0, {}
    dur = end - start
    peak = float(ex.max())
    hook = float(ex[: min(3, len(ex))].max())
    mean = float(ex.mean())
    rel = (int(np.argmax(ex)) + 0.5) / len(ex)
    payoff = float(np.clip((rel - 0.3) / 0.5, 0.0, 1.0))

    sil = sum(max(0.0, min(e, end) - max(s, start)) for s, e in tl.silences)
    silence_ratio = sil / max(dur, 1e-6)
    # 長すぎは強く、短すぎは弱く罰する（中身が濃ければ短くてよい）
    over = max(0.0, dur - cfg.target_duration) / cfg.target_duration
    under = max(0.0, cfg.target_duration - dur) / cfg.target_duration
    dur_pen = over * 0.2 + under * 0.04
    sil_pen = max(0.0, silence_ratio - 0.15) * 0.4

    # 前フリ: 山場（ピークの半分を初めて超える点）より前にどれだけ文脈があるか
    first_hot = int(np.argmax(ex >= peak * 0.5)) if peak > 0 else 0
    setup = min(first_hot / max(cfg.setup_seconds, 1e-6), 1.0)

    # 切り際の判定: 窓の直後(3秒)もまだ盛り上がっていたら「途中で切った」、
    # 直前(3秒)が盛り上がっていたら「別の山の余韻から始めた」
    after = tl.excite[min(tl.n, b):min(tl.n, b + 3)]
    before = tl.excite[max(0, a - 3):a]
    cut_end = float(after.max()) / peak if len(after) and peak > 0 else 0.0
    cut_start = float(before.max()) / peak if len(before) and peak > 0 else 0.0
    cut_pen = 0.30 * max(0.0, cut_end - 0.4) + 0.15 * max(0.0, cut_start - 0.5)

    score = (cfg.w_peak * peak + cfg.w_hook * hook + cfg.w_mean * mean + cfg.w_payoff * payoff
             + 0.06 * setup - dur_pen - sil_pen - cut_pen)
    feats = {
        "cut_end": round(cut_end, 3), "cut_start": round(cut_start, 3),
        "setup": round(setup, 3),
        "peak": round(peak, 3), "hook": round(hook, 3), "mean": round(mean, 3),
        "payoff": round(payoff, 3), "silence_ratio": round(silence_ratio, 3),
    }
    return score, feats


def _steals_other_peak(excite: np.ndarray, start: float, end: float, p: int, radius: int = 5) -> bool:
    """窓が「自分のピークより強い別の山」を含むか。

    含む場合、その窓の点数は他の山からの借り物になり、本命と重複して両方を潰してしまう。
    """
    a, b = int(start), min(len(excite), int(math.ceil(end)))
    own = excite[max(a, p - radius):min(b, p + radius + 1)]
    own_v = float(own.max()) if len(own) else 0.0
    outside = np.concatenate([excite[a:max(a, p - radius)], excite[min(b, p + radius + 1):b]])
    return len(outside) > 0 and float(outside.max()) > own_v * 1.05 + 1e-6


def generate_candidates(segs: list[Segment], tl: Timeline, duration: float, cfg: Config) -> list[Clip]:
    if not segs:
        return _fallback_windows(tl, duration, cfg)
    starts = [s.start for s in segs]
    out: list[Clip] = []
    for p in find_peaks(tl.excite, k=cfg.num_clips * 6):
        k = _seg_at(starts, segs, p + 0.5)
        best: Clip | None = None
        # 終了: ピークを含む文 〜 2文後まで（オチ後のリアクション取り込み）
        for i1 in range(k, min(len(segs), k + 3)):
            if segs[i1].end - p > cfg.max_peak_to_end and i1 > k:
                break
            # 開始: 尺が範囲内になる文頭すべて
            for i0 in range(i1, -1, -1):
                s, e = clip_bounds(segs, i0, i1, cfg, duration)
                d = e - s
                if d > cfg.max_duration:
                    break
                if d < cfg.min_duration:
                    continue
                if _steals_other_peak(tl.excite, s, e, p):
                    continue
                score, feats = score_window(tl, s, e, cfg)
                gap = _gap_before(segs, i0)
                score += min(gap, 1.0) * 0.05           # 間の後から始まる = 話題の切れ目
                if starts_with_connective(segs[i0].text):
                    score -= 0.06                       # 「で、」から始まる = 途中感
                feats["gap_before"] = round(gap, 2)
                if best is None or score > best.score:
                    best = Clip(s, e, float(p), score, feats, (i0, i1))
        if best is None:
            # 発話が少なく尺が足りない場合はピーク中心の固定窓
            s = max(0.0, p - cfg.target_duration * 0.7)
            e = min(duration, s + cfg.target_duration)
            score, feats = score_window(tl, s, e, cfg)
            best = Clip(s, e, float(p), score - 0.05, feats,
                        (_seg_at(starts, segs, s), _seg_at(starts, segs, e)))
        out.append(best)
    return out


def _fallback_windows(tl: Timeline, duration: float, cfg: Config) -> list[Clip]:
    out = []
    for p in find_peaks(tl.excite, k=cfg.num_clips * 6):
        s = max(0.0, p - cfg.target_duration * 0.7)
        e = min(duration, s + cfg.target_duration)
        score, feats = score_window(tl, s, e, cfg)
        out.append(Clip(s, e, float(p), score, feats))
    return out


def overlap(a: Clip, b: Clip) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def select(cands: list[Clip], n: int, thresh: float, min_relative: float = 0.0) -> list[Clip]:
    """スコア順に、既選択と重なりすぎないものを採用（NMS）。1位に比べて弱すぎるものは捨てる。"""
    chosen: list[Clip] = []
    ranked = sorted(cands, key=lambda c: -c.score)
    floor = ranked[0].score * min_relative if ranked and ranked[0].score > 0 else float("-inf")
    # 総合点は前フリ・オチ等の加点で底上げされうるので、山の高さそのものにも下限を設ける
    top_peak = max((c.features.get("peak", 0.0) for c in ranked), default=0.0)
    for c in ranked:
        if c.score < floor:
            break
        if top_peak > 0 and c.features.get("peak", top_peak) < top_peak * min_relative:
            continue
        if all(overlap(c, x) <= thresh for x in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen
