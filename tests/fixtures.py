"""合成テスト素材: 既知の位置に「盛り上がり」を仕込んだ疑似配信を作る。"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 16000

NORMAL = [
    "今日はゆっくり雑談していこうと思います",
    "最近ちょっと寒くなってきましたよね",
    "昨日の配信見てくれた人ありがとう",
    "コメント読んでいきますね",
    "このゲーム前からやりたかったんですよ",
    "じゃあ次のステージ行ってみましょうか",
    "ここは慎重に進めていきたい",
    "アイテムはまだ取っておこうかな",
]
# 正解区間（HIGHLIGHT_START〜HIGHLIGHT_END）で話される内容
HIGHLIGHT = [
    "え、ちょっと待って待って",
    "うそでしょ！？なんでそこにいるの",
    "やばいやばいやばい！！",
    "草www 完全に負けたんだけど",
    "ははははは！もう無理だって",
]
HIGHLIGHT_START = 118.0
HIGHLIGHT_END = 134.0
LONG_SILENCE = (60.0, 61.5)  # ジャンプカット検証用
DECOY = (25.0, 42.0)         # 大音量BGMだけの区間（面白くない）
CHAT_ONLY = (150.0, 160.0)   # 声は普通だがチャットが沸く区間（画面上の出来事）


def _speech(n: int, rng: np.random.Generator, amp: float) -> np.ndarray:
    """音声っぽい信号: 不規則な長さ・強さの音節が並ぶ（実際の声と同じく周期的でない）。"""
    env = np.zeros(n)
    i = 0
    while i < n:
        syl = int(SR * rng.uniform(0.08, 0.28))
        gap = int(SR * rng.uniform(0.0, 0.12))
        a = rng.uniform(0.3, 1.0)
        seg = np.hanning(max(syl, 2)) * a
        env[i:i + syl] = seg[: max(0, min(syl, n - i))]
        i += syl + gap
    t = np.arange(n) / SR
    carrier = 0.6 * np.sin(2 * np.pi * rng.uniform(150, 220) * t) + 0.4 * rng.standard_normal(n)
    return amp * env * carrier


def make_stream(dirpath: Path, duration: float = 180.0, decoy: bool = False
                ) -> tuple[Path, Path, list[tuple[float, float, str]]]:
    rng = np.random.default_rng(0)
    dirpath.mkdir(parents=True, exist_ok=True)
    audio = 0.002 * rng.standard_normal(int(duration * SR))  # 環境ノイズ
    segs: list[tuple[float, float, str]] = []

    t, k = 1.0, 0
    while t < duration - 4:
        in_hl = HIGHLIGHT_START <= t < HIGHLIGHT_END
        if in_hl:
            text = HIGHLIGHT[min(len(HIGHLIGHT) - 1, int((t - HIGHLIGHT_START) / 3.2))]
            dur, amp, gap = 2.8, 0.55, 0.4
        else:
            text = NORMAL[k % len(NORMAL)]
            dur, amp, gap = 3.2, 0.08, 0.5
            k += 1
        nxt = t + dur + gap
        if t < LONG_SILENCE[1] and t + dur > LONG_SILENCE[0]:  # 長い沈黙を確実に作る
            dur = LONG_SILENCE[0] - t
            nxt = LONG_SILENCE[1]
        if dur > 0.5:
            a, b = int(t * SR), int((t + dur) * SR)
            audio[a:b] += _speech(b - a, rng, amp)
            segs.append((round(t, 2), round(t + dur, 2), text))
        t = nxt

    if decoy:  # 大音量の音楽（和音）を重ねる。話し声は普通のまま
        a, b = int(DECOY[0] * SR), int(DECOY[1] * SR)
        tt = np.arange(b - a) / SR
        music = sum(np.sin(2 * np.pi * f * tt) for f in (220, 277, 330, 440)) / 4
        audio[a:b] += 0.6 * music * (0.7 + 0.3 * np.sin(2 * np.pi * 2 * tt))

    wav = dirpath / "audio.wav"
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    video = dirpath / "stream.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={duration}",
         "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )

    srt = dirpath / "stream.srt"
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    srt.write_text("\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(segs, 1)),
                   encoding="utf-8")
    return video, srt, segs


def make_chat(path: Path, duration: float = 180.0) -> Path:
    """ライブチャット (yt-dlp 形式)。HIGHLIGHT と CHAT_ONLY で沸き、他は平常。"""
    import json
    rng = np.random.default_rng(1)
    lines = []
    for t in np.arange(0, duration, 0.5):
        hot = (HIGHLIGHT_START + 3 <= t < HIGHLIGHT_END + 6) or (CHAT_ONLY[0] + 4 <= t < CHAT_ONLY[1] + 4)
        n = rng.poisson(4.0 if hot else 0.4)
        for _ in range(n):
            text = rng.choice(["草", "www", "えぐい！", "やばすぎ"] if hot else ["こんばんは", "初見です", "88888", "かわいい"])
            lines.append(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(int(t * 1000)), "actions": [
                {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]}}}}}]}},
                ensure_ascii=False))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
