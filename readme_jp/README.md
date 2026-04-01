# Deep_Mus_Cut

<p align="center">
  <img src="https://github.com/naomitsu-ozawa/deep_mus_cut/assets/129124821/fae5e681-81a6-409b-923f-0e8e8291d247" />
</p>

![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows%20%7C%20macOS-green)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-AGPL%20v3-pink)

## 動画からマウスの顔をいい感じで画像で保存するアプリ

https://github.com/naomitsu-ozawa/deep_mou_cut_2/assets/129124821/702d32ab-1227-40a7-8f73-65153dc51fd0

## 説明

映像内のマウスの顔を検知していい感じに切り取ってくれます。

## 推奨する使い方

通常の利用では、ブラウザから操作する [`webui/`](../webui) の利用を推奨します。

- 複数動画を GUI でまとめて処理
- ブラウザから結果をダウンロード
- モデル切替と YOLO / CNN しきい値の設定
- モデル確認用の確認動画生成

起動コマンド:

```bash
python webui/app.py
```

通常は標準ブラウザが自動で開きます。  
開かない場合は、以下へアクセスしてください。

```text
http://127.0.0.1:8000
```

詳しくは [`webui/README.md`](../webui/README.md) を参照してください。

## 対応動物

- マウス
  - C57BL/6
  - ICR
  - アカネズミ
  - C3H/He
- 未対応の動物については、お問い合わせください。

---

### Operation Flow

  <details>
    <summary>Pipeline implemented in <code>muscut.py</code></summary>
    <div align="center">
      <img src="https://github.com/user-attachments/assets/51100b51-93c6-4ca1-8a5c-2fb1b32110d4" style="width: 70%; height: auto;" />
    </div>
  </details>
  <details>
    <summary>Pipeline implemented in <code>muscut_with_rembg.py</code></summary>
    <div align="center">
      <img src="https://github.com/user-attachments/assets/67c036d2-5447-46ba-be33-44a688eed670" style="width: 70%; height: auto;" />
    </div>
  </details>

---

## 動作環境

- Ubuntu & Windows
  - CPU : Core i7 　以上
  - GPU : nVidia GPU 　必須
- macOS
  - AppleSilicon 搭載モデル以降

## インストール

- python3.11~で動作します。
- conda 等で仮想環境を作成して下さい。

### Linux(Ubuntu)、Windows(WSL2)

1. リポジトリをクローンします。
   ```
   git clone https://github.com/naomitsu-ozawa/deep_mus_cut.git
   ```
1. リポジトリのディレクトリに移動します。
   ```
   cd deep_mus_cut
   ```
1. env_ubuntu.yml ファイルから仮想環境を構築します。
   ```
   conda env create -f env_ubuntu.yml -n muscut
   ```
1. 作成した仮想環境に入ります。
   ```
   conda activate muscut
   ```
1. 仮想環境に remBG をインストールします。依存関係の競合を避けるため`--no-deps`を付与します。
   ```
   pip install rembg==2.0.53 --no-deps
   ```

## macOS

1. リポジトリをクローンします。
   ```
   git clone https://github.com/naomitsu-ozawa/deep_mus_cut.git
   ```
1. リポジトリのディレクトリに移動します。
   ```
   cd deep_mus_cut
   ```
1. env_macos.yml ファイルから仮想環境を構築します。
   ```
   conda env create -f env_macos.yml -n muscut
   ```
1. 作成した仮想環境に入ります。
   ```
   conda activate muscut
   ```

### 個別インストール（非推奨）

   <details>
      <summary>クリックで開く</summary>
   ### Mac、 Linux、 Windows(WSL2)、共通

1.  リポジトリをクローンします。  
    `git clone https://github.com/naomitsu-ozawa/deep_mus_cut.git`
2.  Ultralytics をインストールします。  
    `pip install ultralytics`
3.  Scikit-learn をインストールしてください。  
    `pip install scikit-learn`
4.  remBG をインストールします。（背景除去を行う場合）  
    `pip install rembg[gpu] `  
    ※GPU が使えなかった場合は、onnxruntime-gpu をチェックしてください。

### Mac

1.  CoreML に対応した Mac の場合は、CoreMLtools をインストールします。  
    `pip install coremltools`
2.  CoreML 非対応の Mac で利用する場合は、Tensorflow をインストールします。  
    `pip install tensorflow`  
    `pip install tensorflow-metal`

- numpy でエラーが起こる場合は、pip の方の numpy を更新します。  
  `pip install -U numpy`

### Linux&Windows(WSL2)

1.  Tensorflow のインストール
    1. Tensorflow は"2.15.x"まで対応しています。（2.16.x~は未対応）
    2. CUDA 対応の Tensorflow をインストールします。
       `pip install 'tensorflow[and-cuda]==2.15.1'`
2.  PyTorch のインストール
    1. tensorflow2.13 以前を使う場合（CUDA11.x を使う場合）
       1. CUDA 対応の PyTorch をインストールするために一度アンインストールします。  
          `pip uninstall torch torchvision torchaudio`  
          こちらから CUDA 対応の PyTorch をインストールします。  
          `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    2. tensorflow2.14 以降を使う場合（CUDA12.x を使う場合）
       1. そのままで大丈夫

   </details>

### アップデート方法

- deep_mus_cut フォルダに移動後、git pull して下さい。

---

## 使い方

### WebUI を使う（推奨）

- WebUI を起動します。

  ```
  python webui/app.py
  ```

- 通常は標準ブラウザが自動で開きます。開かない場合は以下へアクセスしてください。

  ```
  http://127.0.0.1:8000
  ```

- ブラウザ上で以下を行えます。
  - 1 本または複数の動画をアップロード
  - 通常抽出または背景除去あり抽出の実行
  - 抽出結果 zip の個別 / 一括ダウンロード
  - モデルと YOLO / CNN しきい値の変更
  - モデル確認動画の生成

詳しい手順は [`webui/README.md`](../webui/README.md) を参照してください。

### CLI を使う

### シンプルに横顔を収集

- １つの動画ファイルを解析する場合（macOS & Ubuntu）

  - 環境変数"movie"に動画ファイルのパスを格納します。

    ```
    # あなたの準備した動画の場合
    movie="/path/to/your/movie.mov"

    # Demo用の動画を使う場合
    movie="./demo/C57BL6_profile_demo.mp4"
    ```

    ```
    python muscut.py -f $movie -s
    ```

    で解析が始まります。

### 背景を切り抜いた画像の収集

- １つの動画ファイルを解析する場合（macOS & Ubuntu）

  - 環境変数"movie"に動画ファイルのパスを格納します。

    ```
    # あなたの準備した動画の場合
    movie="/path/to/your/movie.mov"

    # Demo用の動画を使う場合
    movie="./demo/C57BL6_profile_demo.mp4"
    ```

    ```
    python muscut_with_rembg.py -f $movie -s
    ```

    で解析が始まります。

- 1 頭の動物で複数の動画を解析する場合（Ubuntu or Windows nVidia GPU が必須）
  - 環境変数"folder"に動画の入ったフォルダーのパスを格納します。
    ```
    folder="/path/to/your/directory"
    ```
    ```
    python batcher_single.py -f folder -ps
    ```
    を実行します。
  - ディレクトリ構造は以下を参考にしてください。
    ```
    動物ごとにフォルダ分けしてください。
    指定したフォルダ内の動画を解析します。
    ├──  animal01 ←　ここを指定する
    │   ├──  C0013.MP4
    │   ├──  C0014.MP4
    │   └──  C0015.MP4
    ├──  animal02
    │   ├──  C0016.MP4
    │   ├──  C0017.MP4
    │   └──  C0018.MP4
    ・
    ・
    ・
    ```
- 複数の動物で大量の動画を解析する場合（Ubuntu or Windows nVidia GPU が必須）
  - 環境変数"folder"に指定した構造のルートフォルダーのパスを格納します。
    ```
    folder="/path/to/your/directory"
    ```
    ```
    python patcher_para.py -f $folder -ps
    ```
    で解析が始まります。
  - ディレクトリ構造は以下を参考にしてください。
    ```
    batcher_single.pyを並列処理しています。
    batcher_singleで指定したフォルダが格納されているルートフォルダを指定します。
    ├── 00_male　←ここを指定する
    │   ├── animal01　←このフォルダごとで並列化
    │   ├── animal02
    │   ├── animal03
    │   ├── animal04
    │   └── animal05
    ```
- 背景付きの画像を保存する場合

  ```
  python muscut.py -f $movie
  ```

- 顔検知中のプレビューを表示させるには、-s オプションをつけて下さい。
  ```
  python muscut.py -f $movie -s
  ```

### 画像の出力先

- 以下のパスに保存されます。
  ```
     deep_mus_cut
        └── croped_image
           └── <動画のファイル名>
              ├── selected_imgs ← muscut.pyを実行した場合
              └── with_rembg ← muscut_with_rembg.pyを実行した場合
  ```

---

### オプション

| option            | description                                                                                                                                                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -f,--file         | 解析したいファイルのパス（必須）[file_path,webcam]<br>-f <file_path>を指定すると動画ファイルの解析を行います。<br>-f webcam0 を指定するとデバイス ID：０のカメラに接続できます。(テスト機能)<br>複数台カメラが接続されている場合は、webcam\*の番号を変更してみて下さい。    |
| -p, --pint        | フォーカスチェックの閾値の指定<br>デフォルト値：2600<br>閾値については[こちら](ピントの閾値について.md)                                                                                                                                                                     |
| -d,--device       | 物体検知部分で利用するデバイス名 [cpu,cuda,mps]<br>--mode tf の時の PyTroch デバイスを指定できます。                                                                                                                                                                        |
| -t,--tool         | 使用するツール名 <br>-t default：未指定と同じ動作になります。<br>-t kmeans_image_extractor：動画から k-means アルゴリズムを利用して指定枚数のフレーム画像を抽出します。<br>-t tf2ml:Tensorflow モデルを CoreML モデルへ変換します。<br>-t sexing (sexing_multi):demo 用<br> |
| -i,--image_format | 出力画像のフォーマット [jpg,png]<br>-i png：デフォルトです。未指定と同じ動作になります。<br>-i jpg：JPEG 形式で保存します。容量を節約したい場合に有効です。                                                                                                                 |
| -s,--show         | プレビューモード                                                                                                                                                                                                                                                            |
| -n,--number       | 抽出枚数                                                                                                                                                                                                                                                                    |
| -wc,--without_cnn | 画像分類を行わずに解析します。※                                                                                                                                                                                                                                             |
| -a,--all          | 検知された画像を全て保存します。k-means は行いません。※                                                                                                                                                                                                                     |

※-wc と-a オプションの組み合わせで、横顔以外の顔画像を取得できます。

- -wc のみ　 → 　横顔以外を含む、検知された全ての画像から K-means 処理を通して取得します。
- -a のみ　 → 　検知された横顔を K-means 処理をせずにすべて取得します。
- -wc -a 　両方の場合　 → 　横顔以外を含む、検知されたすべての画像から K-means 処理を通して取得します。

  Options
  | -wc -a | -wc | -a |
  | --- | --- | --- |
  | 検知されたすべての顔画像を取得 | 検知されたすべての顔画像から指定枚数を取得 | 検知された横顔のみをすべて取得 |

#### mode について

デフォルトは、それぞれのプラットフォームごとで GPU を使うように設定しています。
MacBook 等において明示的に Tensorflow と PyTorch を利用したい場合、指定してください。
| --mode | 詳細 |
| ---- | ---- |
| coreml | 物体検出と画像分類に CoreMl モデルを使用します。(default) |
| tf_pt | 物体検出と画像分類に PyTroch と TensorFlow を使用します。 |

#### device について

モード”tf_pt”時の物体検出で利用する PyTorch デバイスを指定できます。
デフォルトは、それぞれのプラットフォームごとに GPU を使用するように設定しています。明示的に CPU などを使いたい場合に指定してください。
| --device | 詳細 |
| ---- | ---- |
| cpu | 物体検知に cpu を使います。(default) |
| cuda | 物体検知に CUDA を使います。（n Vidia の GPU が必要です。） |
| mps | Apple の Metal Performance Shaders を使います。 |

#### tool について

| --tool                 | 詳細                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------- |
| kmeans_image_extractor | k-means アルゴリズムを使って動画から指定枚数の画像を抽出し、顔の切り取りは行いません。 |
| tf2ml                  | Tensorflow2.x で訓練された CNN を CoreML 形式へ変換します。Mac 専用の機能です。        |
| sexing (sexing_multi)  | 技術 DEMO プログラムです。                                                             |

#### 保存できるフォーマットについて

- オプションを指定しない場合は、png 形式で保存されます。オプションで指定することで jpg 形式で保存可能です。

---

### その他動物への対応

お問い合わせください。

### データセットの公開について

YOLOv8 の学習に使用したデータセットに加え、横顔判定 AI および 雌雄判定 AI の学習データセットも Google Drive に一括で保管しています。
以下のリンクから直接アクセスしてご利用ください。

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Dataset-blue?logo=google-drive)](https://drive.google.com/drive/folders/1ml23h_laIYa0aAGx5i8tLRfHGyTRxWlE?usp=sharing)

## ライセンスについて

このプロジェクトは、Ultralytics YOLOv8（AGPLv3）を使用しており、画像や動画から物体検出を行う機能を備えています。背景除去には RemBG（MIT ライセンス）、画像処理には OpenCV（MIT ライセンス）を使用しています。

このプロジェクト全体は **GNU Affero General Public License v3 (AGPLv3)** に基づいて公開されています。

詳しくは [`LICENSE`](./LICENSE) をご覧ください。

## 使用しているサードパーティライブラリとライセンス

このプロジェクトは以下の外部ライブラリを使用しており、それぞれのライセンスに従っています。

| ライブラリ名                                                     | ライセンス | URL                                        |
| ---------------------------------------------------------------- | ---------- | ------------------------------------------ |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | AGPLv3     | https://github.com/ultralytics/ultralytics |
| [RemBG](https://github.com/danielgatis/rembg)                    | MIT        | https://github.com/danielgatis/rembg       |
| [OpenCV](https://github.com/opencv/opencv)                       | MIT        | https://github.com/opencv/opencv           |

---

## 📄 関連論文

このコードやプロジェクトを使用する場合は、以下の論文を引用してください：

"AI-Driven System for Large-Scale Automated Collection of Mouse Profile Images"  
Naomitsu Ozawa, Yusuke Sakai, Yusuke Sakai, Chihiro Koshimoto, Seiji Shiozawa

bioRxiv 2025.03.27.645835; doi: https://doi.org/10.1101/2025.03.27.645835  
Available at: https://www.biorxiv.org/content/10.1101/2025.03.27.645835v3

### 📚 BibTeX

```
@article{Ozawa2025MouseProfileAI,
  author  = {Naomitsu, Ozawa and Yusuke, Sakai and Yusuke, Sakai and Chihiro, Koshimoto and Seiji, Shiozawa},
  title   = {AI-Driven System for Large-Scale Automated Collection of Mouse Profile Images},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.03.27.645835},
  url     = {https://www.biorxiv.org/content/10.1101/2025.03.27.645835v2}
}

```
