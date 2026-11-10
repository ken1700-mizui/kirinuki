"""音声解析: フレーム音量(dBFS)・沈黙区間・局所的な音量の跳ね。

ffmpeg で 16kHz モノラルにデコードし、ストリーミングで RMS を計算する
（長時間配信でもメモリを食わない）。
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return {
        "duration": float(info["format"].get("duration", 0.0)),
        "has_video": v is not None,
        "has_audio": a is not None,
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
    }


@dataclass
class AudioAnalysis:
    hop: float
    db: np.ndarray  # フレームごとの dBFS

    @property
    def duration(self) -> float:
        return len(self.db) * self.hop

    def silence_threshold(self) -> float:
        if len(self.db) == 0:
            return -40.0
        floor = float(np.percentile(self.db, 10))
        return float(np.clip(floor + 8.0, -50.0, -28.0))

    def silences(self, min_dur: float = 0.3, threshold: float | None = None) -> list[tuple[float, float]]:
        th = self.silence_threshold() if threshold is None else threshold
        quiet = self.db < th
        out = []
        i, n = 0, len(quiet)
        while i < n:
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                if (j - i) * self.hop >= min_dur:
                    out.append((i * self.hop, j * self.hop))
                i = j
            else:
                i += 1
        return out

    def per_second(self, n_sec: int) -> np.ndarray:
        """1秒ごとの平均音量(dB)。パワー平均。"""
        per = int(round(1.0 / self.hop))
        out = np.full(n_sec, -90.0)
        if len(self.db) == 0:
            return out
        power = 10 ** (self.db / 10)
        for t in range(n_sec):
            chunk = power[t * per:(t + 1) * per]
            if len(chunk):
                out[t] = 10 * np.log10(max(chunk.mean(), 1e-9))
        return out


def analyze_audio(path: str | Path, sr: int = 16000, hop: float = 0.05) -> AudioAnalysis:
    hop_n = int(sr * hop)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames: list[np.ndarray] = []
    buf = b""
    frame_bytes = hop_n * 2
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(frame_bytes * 400)
        if not chunk:
            break
        buf += chunk
        n = len(buf) // frame_bytes
        if n:
            arr = np.frombuffer(buf[: n * frame_bytes], dtype=np.int16).astype(np.float32) / 32768.0
            arr = arr.reshape(n, hop_n)
            frames.append(np.sqrt(np.mean(arr * arr, axis=1)))
            buf = buf[n * frame_bytes:]
    proc.wait()
    if proc.returncode != 0 and not frames:
        # 音声トラックなし等
        return AudioAnalysis(hop, np.zeros(0))
    rms = np.concatenate(frames) if frames else np.zeros(0)
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    return AudioAnalysis(hop, db)


def local_jump(x: np.ndarray, window: int = 60, min_spread: float = 3.0, ignore_below: float | None = -85) -> np.ndarray:
    """局所ベースライン（前後window秒の中央値）からの跳ねを 0〜1 に正規化。

    配信全体の差（BGMが大きい区間、視聴者数の増減など）に引っ張られないよう、局所的に比べる。
    """
    n = len(x)
    out = np.zeros(n)
    if n == 0:
        return out
    half = window // 2
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        local = x[lo:hi]
        if ignore_below is not None:
            local = local[local > ignore_below]  # 完全無音は除外
        if len(local) < 3:
            continue
        med = np.median(local)
        mad = np.median(np.abs(local - med)) * 1.4826
        z = (x[t] - med) / max(mad, min_spread)
        out[t] = np.clip(z, 0.0, 3.0) / 3.0
    return out


def loudness_jump(per_sec_db: np.ndarray, window: int = 60) -> np.ndarray:
    return local_jump(per_sec_db, window, min_spread=3.0, ignore_below=-85)
