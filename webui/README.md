# WebUI

`muscut.py` を既存コード無改変でブラウザから呼び出すための最小 WebUI です。

## 方針

- 既存の `muscut.py` は編集しません。
- WebUI 側は別プロセスで `python muscut.py ...` を実行します。
- 実行時のカレントディレクトリはリポジトリ直下に固定します。
- 抽出画像は `croped_image/<アップロード動画名>/selected_imgs` を zip 化して返します。
- 実行中は進捗ページで処理状況とログを定期更新表示します。
- 最大 3 本まで並列実行し、それを超えた処理は待機列に入ります。
- `-p`, `-a`, `-t`, `-wc`, `-dev` に対応しています。

## セットアップ

```bash
cd /home/idm-kurume/ghq/github.com/naomitsu-ozawa/deep_mus_cut
pip install -r webui/requirements.txt
```

## 起動

```bash
cd /home/idm-kurume/ghq/github.com/naomitsu-ozawa/deep_mus_cut
python webui/app.py
```

デフォルトでは `0.0.0.0:8000` で待ち受けます。

## アクセス

サーバPCのIPが `192.168.1.10` の場合:

```text
http://192.168.1.10:8000
```

## 環境変数

```bash
MUSCUT_WEBUI_HOST=0.0.0.0
MUSCUT_WEBUI_PORT=8000
```

## 注意

- 同時実行は最大 3 本です。
- 4 本目以降は順番待ちになり、画面に待機状況が表示されます。
- プレビュー表示 `-s` は使いません。
- `-cl` や webcam 指定は WebUI の用途と合わないため対象外です。
- 依存ライブラリとモデルは既存の `muscut.py` 側の動作条件に従います。
- `-a` を使うと `-n` は付与しません。
