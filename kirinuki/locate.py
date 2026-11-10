"""既存の切り抜き動画が「元配信の何秒目か」を音声で突き止める。

伸びている人間の切り抜きを正解データにするための道具。
切り抜きは BGM・効果音・ジャンプカット・音量正規化が入っているので、
短い窓ごとに音量の包絡（ゲイン不変に正規化）を元配信と相互相関し、一致した位置を投票で集める。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioAnalysis, analyze_audio


@dataclass
class Match:
    start: float          # 元配信での開始秒
    end: float            # 元配信での終了秒
    confidence: float     # 一致した窓の割合 × 平均相関
    windows: list[tuple[float, float, float]]  # (clip_t, stream_t, ncc)


def _feature(db: np.ndarray) -> np.ndarray:
    """音量(dB)から、ゲインや定常BGMに強い特徴（1秒の移動平均からの偏差）を作る。"""
    x = np.maximum(db, -70.0)
    k = 20
    pad = np.pad(x, (k // 2, k - k // 2 - 1), mode="edge")
    base = np.convolve(pad, np.ones(k) / k, mode="valid")
    return x - base


def _ncc_all(stream: np.ndarray, win: np.ndarray, s_fft: np.ndarray, nfft: int,
             csum: np.ndarray, csum2: np.ndarray) -> np.ndarray:
    W = len(win)
    w = win - win.mean()
    sw = np.sqrt((w * w).sum())
    if sw < 1e-6:
        return np.zeros(max(0, len(stream) - W + 1))
    corr = np.fft.irfft(s_fft * np.conj(np.fft.rfft(w, nfft)), nfft)[: len(stream) - W + 1]
    s1 = csum[W:] - csum[:-W]
    s2 = csum2[W:] - csum2[:-W]
    var = np.maximum(s2 - s1 * s1 / W, 1e-9)
    return corr / (np.sqrt(var) * sw)


def locate(clip: str | Path | AudioAnalysis, stream: str | Path | AudioAnalysis,
           win_sec: float = 4.0, step_sec: float = 1.0, min_ncc: float = 0.6) -> Match | None:
    ca = clip if isinstance(clip, AudioAnalysis) else analyze_audio(clip)
    sa = stream if isinstance(stream, AudioAnalysis) else analyze_audio(stream)
    hop = sa.hop
    c, s = _feature(ca.db), _feature(sa.db)
    W, step = int(win_sec / hop), int(step_sec / hop)
    if len(c) < W or len(s) < W:
        return None
    nfft = 1 << int(np.ceil(np.log2(len(s) + W)))
    s_fft = np.fft.rfft(s, nfft)
    csum = np.concatenate([[0.0], np.cumsum(s)])
    csum2 = np.concatenate([[0.0], np.cumsum(s * s)])

    hits = []
    for i in range(0, len(c) - W + 1, step):
        win = c[i:i + W]
        if np.ptp(win) < 3.0:  # ほぼ無音・平坦な窓は情報がない
            continue
        ncc = _ncc_all(s, win, s_fft, nfft, csum, csum2)
        k = int(np.argmax(ncc))
        if ncc[k] >= min_ncc:
            hits.append((i * hop, k * hop, float(ncc[k])))
    if not hits:
        return None

    good = _consistent_chain(hits, max_jump=60.0)
    if not good:
        return None
    start = good[0][1]
    end = good[-1][1] + win_sec
    total_windows = max(1, (len(c) - W) // step + 1)
    conf = len(good) / total_windows * float(np.mean([h[2] for h in good]))
    return Match(start, end, round(min(conf, 1.0), 3), good)


def _consistent_chain(hits: list[tuple[float, float, float]], max_jump: float,
                      tol: float = 0.6) -> list[tuple[float, float, float]]:
    """切り抜きは時間が前に進むだけで、ジャンプカットでは元配信側が先に飛ぶ。
    この性質を満たす（ずれ量 stream_t - clip_t が単調非減少で、飛びが max_jump 以内の）
    窓の列のうち、相関の合計が最大のものを動的計画法で選ぶ。
    """
    hits = sorted(hits)
    n = len(hits)
    best = [h[2] for h in hits]
    prev = [-1] * n
    for j in range(n):
        dj = hits[j][1] - hits[j][0]
        for i in range(j):
            di = hits[i][1] - hits[i][0]
            if -tol <= dj - di <= max_jump and best[i] + hits[j][2] > best[j]:
                best[j], prev[j] = best[i] + hits[j][2], i
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(hits[j])
        j = prev[j]
    return chain[::-1]
