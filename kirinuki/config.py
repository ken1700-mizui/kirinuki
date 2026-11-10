"""調整可能なパラメータ。docs/01 の「編集の具体的なルール」と対応する。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- 尺 ---
    min_duration: float = 15.0
    max_duration: float = 60.0
    target_duration: float = 35.0

    # --- 境界 ---
    lead_in: float = 0.15          # 文頭の何秒前から始めるか
    tail: float = 0.8              # オチ後の余韻
    max_peak_to_end: float = 12.0  # ピークから終了点までの最大距離

    # --- 盛り上がり度の重み ---
    w_loud: float = 0.45
    w_react: float = 0.35
    w_density: float = 0.20
    w_chat: float = 0.35           # チャットがある場合のみ使用
    chat_delay: float = 5.0        # 視聴者の反応遅延（秒）

    # --- 候補スコアの重み ---
    w_peak: float = 0.40
    w_hook: float = 0.25
    w_mean: float = 0.20
    w_payoff: float = 0.15

    # --- 音声 ---
    silence_min: float = 0.3       # 沈黙とみなす最小長
    jump_cut_min: float = 0.6      # これ以上の沈黙を詰める
    jump_cut_keep: float = 0.15    # 詰めた後に残す間

    # --- 選定 ---
    num_clips: int = 5
    min_relative_score: float = 0.5  # 1位の何割未満なら出さない（数合わせの駄作を防ぐ）
    setup_seconds: float = 4.0       # 山場の前に欲しい「前フリ」の秒数
    overlap_threshold: float = 0.3

    # --- 出力 ---
    vertical: bool = True
    width: int = 1080
    height: int = 1920
    fps: int = 30
    font: str = "IPAGothic"
    font_size: int = 76
    max_line_chars: int = 14
    loudness_i: float = -14.0
    teaser: str = "auto"           # auto / on / off
    teaser_max: float = 3.0

    # --- LLM ---
    use_llm: bool = True
    model: str = "claude-opus-5-5"
    llm_candidates: int = 12
    llm_weight: float = 0.6
