# kirinuki — 最強切り抜きAI

長時間の配信・動画から「見どころ」を自動で見つけ、縦型ショート（字幕・タイトル・サムネ付き）として書き出すツール。

```
動画/URL ─▶ 文字起こし ─▶ 音量・笑い語・発話密度・チャットを統合した「盛り上がり度」
        ─▶ 文境界に吸着した候補窓 ─▶ 採点・重複除去 ─▶ (Claude で再ランク・タイトル生成)
        ─▶ ジャンプカット・ティーザー・9:16・字幕・ラウドネス ─▶ clip_XX.mp4 / .jpg / report.md
```

## ドキュメント

1. [切り抜きの基礎](docs/01_切り抜きの基礎.md) — 何が必要で、どんな手順で、どう編集するか
2. [AI化の設計](docs/02_AI化の設計.md) — 調査結果とアルゴリズム
3. [レビューと改善ログ](docs/03_レビューと改善ログ.md) — 5回の改善の記録と今後の課題
4. [評価ループ](docs/04_評価ループ.md) — のりプロで精度を上げ、ホロライブで本番に移る手順

## 必要なもの

- Python 3.10+
- ffmpeg（libx264 / libass 付き）
- 日本語フォント（既定: IPAGothic。`--font` で変更）
- 任意: `faster-whisper`（文字起こし）、`anthropic`（Claude 再ランク）、`yt-dlp`（URL 入力）

```bash
pip install -e ".[whisper,llm,dev]"
```

## 使い方

```bash
# 動画ファイルから（文字起こしは faster-whisper で自動）
python -m kirinuki run 配信.mp4 -o out/

# 既存の字幕とチャットログを使う（精度が上がる）
python -m kirinuki run 配信.mp4 -t 配信.srt -c 配信.live_chat.json -o out/

# URL から（動画・チャット・字幕を yt-dlp で取得）
python -m kirinuki run "https://www.youtube.com/watch?v=..." -o out/

# 検出だけ（書き出さない）
python -m kirinuki analyze 配信.mp4 -t 配信.srt

# 事務所の規約チェック（禁止チャンネル・メン限・誇張タイトル）と概要欄の自動生成
python -m kirinuki run "<URL>" -p noripro

# 評価ループ（詳細は docs/04）
python -m kirinuki refs <伸びている切り抜きURL...> -o refs/
python -m kirinuki truth 元配信.mp4 refs/*.mp4 -o truth.json
python -m kirinuki eval out/report.json truth.json
python -m kirinuki review out/clip_01.mp4 --ref refs/*.mp4
```

Claude 再ランクは `ANTHROPIC_API_KEY` を設定すると自動で有効になります（`--no-llm` で無効）。
既定モデルは `claude-opus-5-5` です。拒否時のサーバー側フォールバック（`fallbacks: "default"`）を有効にしています。

主なオプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `-n` | 5 | 最大クリップ数（品質下限未満は出さない） |
| `--min/--max/--target` | 15/60/30 | 尺（秒） |
| `--teaser` | auto | 山場を冒頭にチラ見せ（auto/on/off） |
| `--horizontal` | - | 横型 16:9 で出力 |
| `--min-relative` | 0.5 | 1位の何割未満のクリップを捨てるか |

## 出力

- `clip_XX.mp4` — 1080x1920, H.264/AAC, -14 LUFS
- `clip_XX.jpg` — サムネイル
- `clip_XX.txt` — 概要欄（元配信URL・タイムスタンプ・クレジット）
- `report.md` / `report.json` — 区間・スコア・タイトル・理由・規約の注意・投稿前チェックリスト

## テスト

```bash
python -m pytest -q
```

合成配信（正解区間・大音量BGMの罠・チャットだけの山場を仕込んだもの）で検出精度を回帰テストしています。

## 注意

切り抜きの可否・収益化・クレジット表記は配信者ごとに異なります。**投稿前に必ず公式の切り抜きガイドラインを確認してください。**
