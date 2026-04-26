# ToolForLogo

`ToolForLogo` は、製品名と説明から多数のロゴ方向性を出して、比較し、残し、派生させ、書き出すためのローカル中心ワークベンチです。

今回の構成は `TimelineForAudio` / `TimelineForVideo` を参考に、単一コンテナではなく次の 2 層に切り替えています。

- `web`: 設定画面、案件作成、案件一覧、候補ギャラリー、export 操作
- `worker`: Python 実行、local model ダウンロード、候補生成 job の実処理

重要:

- `ComfyUI` は使いません
- 画像生成は worker 側の local model backend を前提にします
- ロゴのマーク生成を model が担当し、ワードマークと lockup は ToolForLogo 側で合成します

## いまの到達点

できること:

- Settings で compute mode / quality / Hugging Face token / model preset を管理
- preset ごとの download / delete / cache clear を worker job として実行
- 新規案件を作成して 20-30 件の image exploration batch を queue
- 案件一覧で job の進捗を確認
- 案件詳細で候補の preview を見て複数選択で favorite / adopted / excluded を更新
- favorite / adopted を軸に export bundle を作成
- `comparison_sheet.png` と ZIP export を生成
- CLI から settings / models / jobs / daemon を操作

未実装またはこれから詰めるもの:

- 小サイズ視認性スコア
- 禁止モチーフの構造化入力
- より細かい方向性プリセット
- GPU overlay compose

## アーキテクチャ

```text
ToolForLogo/
  configs/
    runtime.defaults.json
  docker/
    web.Dockerfile
    worker.Dockerfile
  src/tool_for_logo/
    backends.py
    cli.py
    generator.py
    job_store.py
    model_catalog.py
    settings.py
    state.py
    web_app.py
    worker.py
    templates/
    static/
  tests/
  docker-compose.yml
  .env.example
  start.bat
  stop.bat
```

shared roots:

- app data: `C:\Codex\workspaces\ToolForLogo\app-data`
- uploads: `C:\Codex\workspaces\ToolForLogo\uploads`
- outputs / jobs: `C:\Codex\workspaces\ToolForLogo\outputs`
- reports: `C:\Codex\reports\ToolForLogo`
- archive: `C:\Codex\archive\ToolForLogo`
- Hugging Face cache: `C:\Codex\workspaces\ToolForLogo\cache\huggingface`
- Torch cache: `C:\Codex\workspaces\ToolForLogo\cache\torch`

## Backend

現在の backend:

- `diffusers`: worker container 内で local model を直接ロードして 30 案前後の logo exploration mark を生成する標準 backend
- `local-svg`: ローカル LLM で spec を作り、SVG / preview を組み立てる backend
- `mock`: モデル未導入でも flow を確認できる描画 backend

標準 preset:

- `cpu-standard`: `segmind/tiny-sd`
- `gpu-standard`: `stabilityai/sdxl-turbo`
- `gpu-high`: `stabilityai/stable-diffusion-xl-base-1.0`

production dependency note:

- `peft` を worker に追加しています。理由は SDXL base に logo 専用 LoRA を載せて、汎用 `diffusers` 単体よりロゴ探索の当たり率を上げるためです。

`Settings` から download して cache を持てます。`allow auto download` を有効にすると、未配置時に worker が自動取得します。

## Quick Start

Windows の主導線:

```powershell
.\start.bat
```

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

## CLI

web を起動:

```powershell
$env:PYTHONPATH="src"
python -m tool_for_logo web --host 127.0.0.1 --port 19130
```

worker daemon:

```powershell
$env:PYTHONPATH="src"
python -m tool_for_logo daemon --poll-interval 5
```

settings:

```powershell
python -m tool_for_logo settings status --json
python -m tool_for_logo settings save --compute-mode cpu --processing-quality standard --default-batch-count 10 --json
```

models:

```powershell
python -m tool_for_logo models list --json
python -m tool_for_logo models download --preset-id cpu-standard --json
python -m tool_for_logo models delete --preset-id cpu-standard --json
```

jobs:

```powershell
python -m tool_for_logo jobs list --json
python -m tool_for_logo jobs create-batch --case-id <case_id> --count 10 --direction-hint "premium editorial" --json
python -m tool_for_logo jobs run --job-id <job_id> --json
```

直接 batch 実行:

```powershell
python -m tool_for_logo create-case --name "ToolForLogo" --description "Local-first logo exploration workspace" --json
python -m tool_for_logo generate-batch --case-id <case_id> --count 4 --backend mock --json
python -m tool_for_logo export --case-id <case_id> --json
```

## Settings 画面の役割

`TimelineForAudio` / `TimelineForVideo` と同じ考え方で、Settings は次をまとめる入口です。

- compute mode
- processing quality
- Hugging Face token
- active model preset
- cache usage
- model download / delete

## データの持ち方

案件データ:

- `app-data/cases/<case_id>/case.json`
- `app-data/cases/<case_id>/batches/*.json`
- `app-data/cases/<case_id>/candidates/<candidate_id>/candidate.json`
- `app-data/cases/<case_id>/exports/*.json`

worker jobs:

- `outputs/jobs/<job_id>/request.json`
- `outputs/jobs/<job_id>/status.json`
- `outputs/jobs/<job_id>/result.json`
- `outputs/jobs/<job_id>/worker.log`

export:

- `C:\Codex\reports\ToolForLogo\<case_id>\<export_id>\manifest.json`
- `C:\Codex\reports\ToolForLogo\<case_id>\<export_id>\comparison_sheet.png`
- `C:\Codex\reports\ToolForLogo\<case_id>\<export_id>.zip`

## テスト

```powershell
$env:PYTHONPATH="src"
pytest tests
```
