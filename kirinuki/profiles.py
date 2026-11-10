"""事務所ごとの規約設定: 投稿可否のチェック・タイトルの誇張チェック・概要欄の自動生成。

※ ここに書いたルールは調査時点の要約。必ず公式ガイドラインの最新版で確認すること。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Clip

# 誤解・誇張を招きやすい語（多くの事務所で「誤解を招くタイトル/サムネ」は禁止）
EXAGGERATION = ["炎上", "暴露", "閲覧注意", "衝撃", "悲報", "ブチギレ", "大炎上", "引退", "卒業",
                "裏の顔", "本性", "放送事故", "BAN", "晒", "喧嘩", "不仲", "前世", "中の人"]
MEMBERS_ONLY = ["メン限", "メンバー限定", "members only", "membership", "メンバーシップ限定"]
KARAOKE = ["歌枠", "karaoke", "カラオケ", "歌ってみた", "3Dライブ", "LIVE", "ライブ"]


@dataclass
class Profile:
    key: str
    name: str
    guideline_url: str
    # 切り抜き禁止（または公認のみ）のチャンネル/タレント名
    blocked_channels: list[str] = field(default_factory=list)
    block_karaoke: bool = True
    talents: dict[str, str] = field(default_factory=dict)  # タレント名 -> メモ
    description_header: str = ""
    notes: list[str] = field(default_factory=list)


PROFILES: dict[str, Profile] = {
    "noripro": Profile(
        key="noripro",
        name="のりプロ",
        guideline_url="https://noripro.jp/fanart/",
        blocked_channels=["犬山たまき", "Tamaki Ch", "佃煮のりお"],
        talents={
            "斎木こまり": "雑談・お絵描き・作業配信。作業配信(無言)は切り抜き不向き",
            "レグルシュ・ライオンハート": "ゲーム実況・雑談・食レポ。リアクション型",
        },
        description_header="",
        notes=[
            "概要欄に元動画のURLを必ず記載",
            "タレント個別ルールを公式ガイドラインで確認（随時追加される）",
            "他事務所のタレントが出演している場合は、その人のルールも守る",
        ],
    ),
    "hololive": Profile(
        key="hololive",
        name="ホロライブ",
        guideline_url="https://hololivepro.com/terms/",
        description_header="【元配信】{title}\n{url}\n",
        notes=[
            "切り抜きチャンネルの事前登録が必要",
            "概要欄の冒頭に元配信のタイトルとURL",
            "メンバー限定・有料配信は不可、アーカイブ公開前の投稿は不可",
            "法人による切り抜きは不可",
        ],
    ),
}


def get(key: str | None) -> Profile | None:
    if not key:
        return None
    if key not in PROFILES:
        raise ValueError(f"未知のプロファイル: {key}（{', '.join(PROFILES)}）")
    return PROFILES[key]


def _contains(text: str, words: list[str]) -> list[str]:
    low = text.lower()
    return [w for w in words if w.lower() in low]


def check_source(p: Profile, info: dict) -> tuple[list[str], list[str]]:
    """元配信が切り抜けるか。(errors, warnings) を返す。errors があれば処理を止める。"""
    errors, warnings = [], []
    channel = " ".join(str(info.get(k, "")) for k in ("channel", "uploader", "uploader_id"))
    title = str(info.get("title", ""))
    for b in p.blocked_channels:
        if b and b in channel:
            errors.append(f"「{b}」のチャンネルは切り抜き禁止/公認のみです")
    if info.get("availability") in ("subscriber_only", "premium_only", "needs_auth") or _contains(title, MEMBERS_ONLY):
        errors.append("メンバー限定・有料配信は切り抜きできません")
    if info.get("live_status") in ("is_live", "is_upcoming"):
        errors.append("配信中/未公開の配信は切り抜けません（アーカイブ公開後に）")
    if p.block_karaoke and _contains(title, KARAOKE):
        warnings.append("歌枠・ライブの可能性: 歌唱部分は権利上切り抜き不可のことが多い")
    if not info:
        warnings.append("元配信のメタデータが無いため、規約チェックを一部省略しました")
    return errors, warnings


def check_title(title: str) -> list[str]:
    hits = _contains(title, EXAGGERATION)
    return [f"誇張・誤解を招く恐れのある語: {', '.join(hits)}"] if hits else []


def _ts(t: float) -> str:
    t = int(t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def make_description(p: Profile | None, info: dict, clip: Clip) -> str:
    url = info.get("webpage_url") or info.get("original_url") or ""
    title = info.get("title", "")
    channel = info.get("channel") or info.get("uploader") or ""
    stamp = f"{url}{'&' if '?' in url else '?'}t={int(clip.start)}s" if url else ""
    head = p.description_header.format(title=title, url=url) if p and p.description_header else ""
    lines = [head.rstrip()] if head else []
    lines += [
        f"{clip.title}" if clip.title else "",
        "",
        f"▼元配信（{_ts(clip.start)}〜）",
        f"{title}" if title else "",
        stamp or "（元配信URLを記入してください）",
        "",
        f"▼配信者: {channel}" if channel else "",
    ]
    if p:
        lines += ["", f"※{p.name}の二次創作ガイドラインに基づいて制作しています", p.guideline_url]
    text = "\n".join(x for x in lines if x is not None).strip()
    return re.sub(r"\n{3,}", "\n\n", text) + "\n"
