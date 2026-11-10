"""テキストシグナル: 笑い・驚き・感情語、発話密度、文頭の接続詞判定。"""
from __future__ import annotations

import re

# (パターン, 重み)。日本語配信の切り抜きで実際に盛り上がりのサインになる語。
REACTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"草|くさ生え|大草原"), 1.0),
    (re.compile(r"(?<![A-Za-z])[wｗ]{2,}|(?<=[ぁ-んァ-ン一-龥！!？?])[wｗ](?![A-Za-z])"), 1.0),
    (re.compile(r"[（(]笑[)）]|爆笑|笑笑|ワロタ|わろた"), 1.0),
    (re.compile(r"(?:あは|はは|ハハ|ふふ|へへ|ひひ|ヒヒ|ぶはっ|ぎゃは){2,}|ははは|あはは"), 0.9),
    (re.compile(r"やば|ヤバ|ヤバい|やべ|ヤベ"), 0.7),
    (re.compile(r"マジ|まじで|ガチ"), 0.4),
    (re.compile(r"うそ|ウソ|嘘でしょ|ありえな|有り得な"), 0.7),
    (re.compile(r"なんで|何で|どういうこと|は[？?]"), 0.5),
    (re.compile(r"すご|凄|天才|神(?:回|プレイ|すぎ)|最強|えぐ|エグ"), 0.6),
    (re.compile(r"ちょっと待って|待って待って|まってまって|やめて|ダメダメ|だめだめ"), 0.7),
    (re.compile(r"きた[ぁあー]*[!！]|キタ[ァアー]*|よっしゃ|しゃあ|勝った|やった[ぁあー]*[!！]"), 0.8),
    (re.compile(r"([ぁ-んァ-ンー])\1{3,}"), 0.6),  # ああああ / ぎゃああああ
    (re.compile(r"[!！]{2,}|[?？][!！]|[!！][?？]"), 0.6),
    (re.compile(r"[!！]"), 0.25),
    (re.compile(r"泣|涙|エモ|感動"), 0.5),
    (re.compile(r"\b(?:lol|lmao|haha+|omg|wtf|no way|insane|holy)\b", re.I), 0.8),
]

# 文頭にあると「話の途中から始まった」感が出る接続詞。
CONNECTIVES = re.compile(
    r"^(?:で、|でさ|それで|だから|なので|でも|けど|けれど|あと|んで|ってか|つまり|そしたら|"
    r"and\b|but\b|so\b|because\b)",
    re.I,
)

_CJK = re.compile(r"[ぁ-んァ-ン一-龥ー]")


def reaction_score(text: str) -> float:
    """0〜1。複数の語が重なるほど高い（上限あり）。"""
    total = 0.0
    for pat, w in REACTION_PATTERNS:
        n = len(pat.findall(text))
        if n:
            total += w * min(n, 3) ** 0.5
    return min(total / 2.5, 1.0)


def reaction_words(text: str) -> list[str]:
    found: list[str] = []
    for pat, _ in REACTION_PATTERNS:
        for m in pat.finditer(text):
            w = m.group(0)
            if len(w.strip("!！?？")) >= 1 and w not in found:
                found.append(w)
    return found


def char_count(text: str) -> int:
    """発話量の指標。日本語は1文字=1、英語は単語数×2 で近似。"""
    cjk = len(_CJK.findall(text))
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    return cjk + latin_words * 2


def starts_with_connective(text: str) -> bool:
    return bool(CONNECTIVES.match(text.strip()))
