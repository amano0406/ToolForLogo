# ToolForLogo

`ToolForLogo` は、サービス名と説明からロゴ案を大量に比較し、残し、派生させ、名前差し替え付きで持ち出すためのローカル中心 CLI / HTTP ツールです。UI はまだ入れず、案件・候補・バッチ・書き出しの運用契約を先に固定しています。

このプロダクトの主 backend は、既存製品 `C:\apps\ComfyUI_windows_portable` を使う `comfyui` です。`ToolForLogo` 自身は、AI が出した図形をそのまま見せるのではなく、ローカルでマーク抽出、配色の整形、ワードマーク合成、比較シート生成まで担当します。

## 位置づけ

- Dock 系ではありません
- Timeline 系でもありません
- 新しい `tool` 系として扱います

理由:

- 主目的が外部サービス操作ではなく、ローカルでの案探索と選定だから
- 画像 1 枚の生成器ではなく、案件単位の比較と仕上げが中心だから
- 将来 UI を乗せる前に CLI と state contract を固めたいから

## Backend

現在の backend:

- `comfyui`: 既存のローカル ComfyUI API を使ってマークの原案を生成
- `mock`: 実画像生成なしでフロー確認
- `openai`: 任意のクラウド fallback。標準依存には含めていません

標準運用では `TOOL_FOR_LOGO_DEFAULT_BACKEND=comfyui` を使います。

## 入力

- 製品名
- 製品説明
- 任意の方向性ヒント
- 任意の seed
- 任意の派生元候補
- 任意の name override

## 出力

- 案件 state: `C:\Codex\workspaces\ToolForLogo`
- レポート / 比較シート / export bundle: `C:\Codex\reports\ToolForLogo`
- 長期退避先の予約: `C:\Codex\archive\ToolForLogo`

候補ごとに次を保存します。

- `mark.png`
- `mark_raw.png` (`comfyui` / `openai` のときのみ)
- `mark.svg` (`mock` のときのみ)
- `wordmark.png`
- `wordmark.svg`
- `lockup_horizontal.png`
- `lockup_stacked.png`
- `preview_light.png`
- `preview_dark.png`
- `candidate.json`

書き出し時は `manifest.json`、`comparison_sheet.png`、ZIP を生成します。

## Quick Start

Windows の主導線:

```powershell
.\start.bat
```

`start.bat` の責務:

1. Docker Desktop / engine の確認
2. `.env` 自動生成
3. `C:\Codex\...` 配下の出力先作成
4. ComfyUI API の readiness 確認
5. 必要なら `C:\apps\ComfyUI_windows_portable\run_nvidia_gpu_api.bat` を起動
6. `docker compose up --build -d`
7. `/health` で起動確認

停止:

```powershell
.\stop.bat
```

手動起動:

```powershell
docker compose up --build -d
```

health:

```text
http://127.0.0.1:19130/health
```

## Requirements

- Windows + Docker Desktop
- 既存の `C:\apps\ComfyUI_windows_portable`
- Python 3.11+ でローカル CLI 実行も可能

標準 production dependency:

- `Pillow`

依存理由:

- `Pillow`: マーク整形、ワードマーク合成、preview、比較シート生成のため

任意 dependency:

- `openai`: `backend=openai` を明示的に使いたいときだけ手動追加

## `.env`

初回起動時に `.env.example` から `.env` を自動生成します。

主な値:

- `TOOL_FOR_LOGO_WEB_PORT`
- `TOOL_FOR_LOGO_HOST_STATE_ROOT`
- `TOOL_FOR_LOGO_HOST_REPORT_ROOT`
- `TOOL_FOR_LOGO_HOST_ARCHIVE_ROOT`
- `TOOL_FOR_LOGO_STATE_ROOT`
- `TOOL_FOR_LOGO_REPORT_ROOT`
- `TOOL_FOR_LOGO_ARCHIVE_ROOT`
- `TOOL_FOR_LOGO_DEFAULT_BACKEND`
- `TOOL_FOR_LOGO_START_COMFYUI`
- `TOOL_FOR_LOGO_COMFYUI_DIR`
- `TOOL_FOR_LOGO_COMFYUI_BASE_URL`
- `TOOL_FOR_LOGO_COMFYUI_CHECKPOINT`
- `TOOL_FOR_LOGO_COMFYUI_WIDTH`
- `TOOL_FOR_LOGO_COMFYUI_HEIGHT`
- `TOOL_FOR_LOGO_COMFYUI_STEPS`
- `TOOL_FOR_LOGO_COMFYUI_CFG`
- `TOOL_FOR_LOGO_COMFYUI_SAMPLER`
- `TOOL_FOR_LOGO_COMFYUI_SCHEDULER`
- `TOOL_FOR_LOGO_COMFYUI_TIMEOUT_SECONDS`
- `TOOL_FOR_LOGO_COMFYUI_POLL_SECONDS`
- `TOOL_FOR_LOGO_COMFYUI_NEGATIVE_PROMPT`

## API Surface

- `GET /health`
- `GET /api/status`
- `GET /api/cases`
- `POST /api/cases`
- `GET /api/cases/{case_id}`
- `POST /api/cases/{case_id}/batches`
- `POST /api/cases/{case_id}/candidates/{candidate_id}/status`
- `POST /api/cases/{case_id}/exports`

## CLI

ローカル実行例:

```powershell
$env:PYTHONPATH="src"
python -m tool_for_logo create-case --name "Northwind Atlas" --description "Global logistics orchestration platform"
python -m tool_for_logo generate-batch --case-id "<case_id>" --count 20 --backend comfyui --direction-hint "premium editorial"
python -m tool_for_logo set-status --case-id "<case_id>" --candidate-id "<candidate_id>" --status favorite
python -m tool_for_logo generate-batch --case-id "<case_id>" --count 6 --from-candidate "<candidate_id>" --backend comfyui
python -m tool_for_logo export --case-id "<case_id>" --name-override "Northwind One"
python -m tool_for_logo status --json
```

`mock` でフローだけ確認したい場合:

```powershell
python -m tool_for_logo generate-batch --case-id "<case_id>" --count 8 --backend mock
```

`openai` を使いたい場合は、自分で `pip install openai` したうえで `--backend openai` を指定します。

## 利用フロー

1. 案件を作る
2. まず 20 案前後を `comfyui` または `mock` backend で一括生成する
3. `favorite` / `excluded` / `adopted` を付ける
4. 気に入った候補から `--from-candidate` で派生案を追加する
5. 必要なら製品名だけ差し替えて export する
6. `comparison_sheet.png` で比較する

## State Layout

```text
ToolForLogo/
  docker/
  src/tool_for_logo/
  tests/
  .env.example
  docker-compose.yml
  start.bat
  stop.bat
  README.md

C:\Codex\workspaces\ToolForLogo\
  state\
    cases\
      <case_id>\
        case.json
        batches\
        candidates\
        exports\
    logs\

C:\Codex\reports\ToolForLogo\
  <case_id>\
    <export_id>\
      manifest.json
      comparison_sheet.png
      <candidate_id>\
        assets\
    <export_id>.zip
```

## 概念モデル

- `case`: 1 つのサービスやプロダクトに対するロゴ検討単位
- `batch`: 1 回の一括生成
- `candidate`: 各ロゴ案
- `status`: `fresh` / `favorite` / `excluded` / `adopted`
- `export`: 持ち出し用にまとめた成果物

## 今回のスコープ

- 案件管理
- ローカル ComfyUI による実マーク生成
- mock 一括生成
- 派生バッチ
- マークと文字の分離保存
- AI 原案からの背景除去とブランド配色への整形
- 横組み / 縦組みの lockup 生成
- 背景明暗 preview
- `comparison_sheet.png`
- name override export
- Docker 一発起動
- `GET /health`
- CLI `status`

## 未実装

- 比較 UI / ギャラリー UI
- フォント保存
- 禁止モチーフ指定
- 小サイズ視認性の自動採点
- 方向性の言い換え再展開

## テスト

```powershell
$env:PYTHONPATH="src"
pytest tests
```
