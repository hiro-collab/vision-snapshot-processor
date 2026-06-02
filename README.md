# Vision Snapshot Processor

Camera Hub / MediaMTX の映像 stream から低頻度 snapshot を取り、軽量な vision state を
Camera Hub topic envelope 互換の WebSocket topic として配信する独立 module です。

## Role

- MediaMTX の RTSP stream を読む。
- snapshot 系 processor を実行する。
- `/vision/.../state` topic を WebSocket で配信する。

この module は物理カメラを直接開きません。Environment State Server への HTTP 書き込みや
Dify 向け snapshot 集約も担当しません。

## Initial Processor

| Processor | Topic | Purpose |
|---|---|---|
| `room_light` | `/vision/room_light/state` | 室内照明が点灯している可能性 |

将来は同じ runtime に、在室、窓・カーテン状態、特定物体状態などの snapshot processor を追加できます。

## 初期セットアップ

```powershell
uv sync
uv run python -m unittest discover -s tests
```

通常運用では、先に USB camera を FFmpeg / MediaMTX / Camera Hub 側で配信しておきます。この organ は
RTSP stream を低頻度で読み、snapshot state topic を出すだけです。

## dotenv / local config

標準の `.env` は不要です。接続先は起動引数で渡します。RTSP URL、frame id、processor 名を
環境に合わせて指定してください。

snapshot、debug frame、`.cache/`、`.venv/` はローカル資材です。Git に含めません。

## 通常起動

```powershell
cd <workspace>\vision-snapshot-processor
uv run python -m vision_snapshot_processor.main `
  --host 127.0.0.1 `
  --port 8776 `
  --camera-source rtsp://127.0.0.1:8554/cam0 `
  --frame-id cam0 `
  --processor room_light
```

通常は Home Control Stack から起動します。Environment State Server はこの WebSocket を
追加 vision topic source として購読します。

## Room Light Heuristic Scope

初期実装は重い学習モデルを使わない conservative heuristic です。Logitech などの自動明るさ補正を
前提に、単純な平均輝度ではなく、色度、彩度、局所コントラスト、暗部/白飛び比率、時系列差分を
時間窓で集約します。daylight が強い場合は電気 OFF と断定せず `unknown` に寄せます。
