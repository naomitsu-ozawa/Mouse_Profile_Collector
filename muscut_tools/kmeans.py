# tsne version cluster center
import glob
import logging
import multiprocessing
import os
import random
import re
import shutil
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    from umap import UMAP
except ImportError:
    UMAP = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import tensorflow as tf
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D


# t-SNE/UMAPの2次元散布図による可視化
def plot_tsne2d(df, width=800, height=800, axis_label_prefix="TSNE"):
    figsize = (width / 96, height / 96)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df, x="TSNE1", y="TSNE2", hue="label", palette="bright", legend="full"
    )
    ax.set_xlabel(f"{axis_label_prefix}1")
    ax.set_ylabel(f"{axis_label_prefix}2")

    # クラス番号を各クラスのポイント群の中心にプロット
    labels = df["label"].unique()
    colors = sns.color_palette("tab10", len(labels))
    for label in labels:
        subset = df[df["label"] == label]
        center_x = subset["TSNE1"].mean()
        center_y = subset["TSNE2"].mean()
        ax.text(
            center_x,
            center_y,
            str(label),
            fontsize=12,
            weight="bold",
            color="black",
            ha="center",
            va="center",
        )

    # plt.show()

    return fig, ax


# t-SNE/UMAPの3次元散布図による可視化
def plot_tsne3d(df, width=800, height=800, axis_label_prefix="TSNE"):
    figsize = (width / 96, height / 96)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # 軸にラベルを付ける
    ax.set_xlabel(f"{axis_label_prefix}1")
    ax.set_ylabel(f"{axis_label_prefix}2")
    ax.set_zlabel(f"{axis_label_prefix}3")

    # 散布図を描画
    labels = df["label"].unique()
    for label in labels:
        ax.scatter(
            df.loc[df["label"] == label, "TSNE1"],
            df.loc[df["label"] == label, "TSNE2"],
            df.loc[df["label"] == label, "TSNE3"],
            alpha=0.8,
            marker=".",
            label=label,
        )

    # クラス番号を各クラスのポイント群の中心にプロット
    for label in labels:
        subset = df[df["label"] == label]
        center_x = subset["TSNE1"].mean()
        center_y = subset["TSNE2"].mean()
        center_z = subset["TSNE3"].mean()
        ax.text(
            center_x,
            center_y,
            center_z,
            str(label),
            fontsize=12,
            weight="bold",
            color="black",
        )

    plt.legend()
    # plt.show()

    return fig, ax


# tqdm用　アップデート関数
def update(num, ax, pbar):
    ax.view_init(elev=10.0, azim=num)
    pbar.update(1)


# kmeansのモデル構築
def build_kmeans(df, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, n_init="auto")
    """
    n_initのデフォルトは10
    50でいい感じになった
    n_init="auto" => k-means++になる
    """
    kmeans.fit(df)

    return kmeans


# エルボー法のグラフ作成用
def test_kmeans(df):
    test_sse = []
    t_range = math.ceil(len(df) / 2)
    test_kmeans_pbar = tqdm(range(1, t_range))

    for k in test_kmeans_pbar:
        kmeans_t = KMeans(n_clusters=k, random_state=0)
        kmeans_t.fit(df)
        test_sse.append(kmeans_t.inertia_)
    return test_sse


# t-SNEのモデル構築
def build_tsne(df, n_components=2):
    tsne = TSNE(
        n_components=n_components,
        # random_state=0,
        perplexity=26,  # 26
        init="pca",
    )
    """
    perplexity
    小さいほど、局所的な構造に焦点を当てる
    大きいほど、大域的な構造に焦点を当てる
    """
    tsne_results = tsne.fit_transform(df)
    return tsne_results


# UMAPのモデル構築
def build_umap(df, n_components=2):
    if UMAP is None:
        raise ImportError(
            "UMAP is not available. Please install `umap-learn` to use reduction_method='umap'."
        )
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    umap_results = umap_model.fit_transform(df)
    return umap_results


def build_embedding(df, method="tsne", n_components=3):
    method_lower = method.lower()
    if method_lower == "tsne":
        return build_tsne(df, n_components=n_components)
    if method_lower == "umap":
        return build_umap(df, n_components=n_components)
    raise ValueError(f"Unsupported reduction method: {method}")


# k-meansをもとに画像のファイル名を変更して保存する関数
def process_images(
    start,
    end,
    imgs_list,
    save_path,
    kmeans,
    format_flag,
    video_name,
    for_kmeans_frame_no,
    progress_queue,
    idx_list,
):
    for j in range(start, end):
        label = kmeans.labels_[j]
        img = imgs_list[j]
        # file_number = str(frame_no).zfill(6)
        # file_number = for_kmeans_frame_no[j].zfill(6)
        file_number = str(for_kmeans_frame_no[j]).zfill(6)

        idx = idx_list[j]
        if format_flag == "jpg":
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(
                    label, f"class{label}_{video_name}-{file_number}_idx{idx}.jpg"
                ),
                img,
            )
        else:
            cv2.imwrite(
                save_path
                + "cluster{}/{}".format(
                    label, f"class{label}_{video_name}-{file_number}_idx{idx}.png"
                ),
                img,
            )
    progress_queue.put(1)  # プロセスの処理完了をキューに追加


# process imagesを並列処理
def make_cluster_dir_parallel(
    imgs_list, save_path, kmeans, format_flag, video_name, for_kmeans_frame_no, idx_list
):
    # 保存先のディレクトリを空にして作成
    shutil.rmtree(save_path)
    os.mkdir(save_path)
    # クラスタごとのディレクトリ作成
    for i in tqdm(range(kmeans.n_clusters), desc="Creating directories"):
        cluster_dir = save_path + "cluster{}".format(i)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

    num_processes = multiprocessing.cpu_count()
    processes = []
    imgs_per_process = len(imgs_list) // num_processes
    progress_queue = multiprocessing.Queue()  # プロセスの進捗を管理するためのキュー

    for i in range(num_processes):
        start = i * imgs_per_process
        end = start + imgs_per_process if i < num_processes - 1 else len(imgs_list)
        p = multiprocessing.Process(
            target=process_images,
            args=(
                start,
                end,
                imgs_list,
                save_path,
                kmeans,
                format_flag,
                video_name,
                for_kmeans_frame_no,
                progress_queue,
                idx_list,
            ),
        )
        processes.append(p)
        p.start()

    # プロセスの処理完了を監視し、進捗を更新する
    for _ in tqdm(range(num_processes), desc="Processing images"):
        progress_queue.get()

    for p in processes:
        p.join()

    print("\033[32mクラスタごとにファイル作成完了\033[0m")


# 画像から特徴量を抽出してリストを返す
def create_npy_image_list(for_kmeans_array, kmeans_cnn):
    # def create_npy_image_list(for_kmeans_array):
    npy_image_list = []
    # print("\033[32m配列変換中・・・\033[0m")
    print("\033[32mMobilenetで特徴量抽出中・・・\033[0m")
    #########################################################
    # 画像をバッチで読むバージョン
    try:
        img_npys = [cv2.resize(img_npy, (224, 224)) for img_npy in for_kmeans_array]
        img_npys = [cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB) for img_npy in img_npys]
        img_npys = np.stack(img_npys, axis=0)
        preds = kmeans_cnn.predict(img_npys, verbose=0)
        # npy_image_list = [pred.flatten() for pred in preds]
        npy_image_list = [
            ((pred.flatten() - pred.min()) / (pred.max() - pred.min()))
            for pred in preds
        ]
        # print(npy_image_list[0])

    except:
        print("cnn future extract error")

    print("\033[32m抽出完了\033[0m")
    return npy_image_list


# k-means処理
def kmeans_main(
    save_path,
    video_name,
    for_kmeans_array,
    cluster_num,
    format_flag,
    for_kmeans_frame_no,
    kmeans_cnn,
    dev_flag=False,
    reduction_method="tsne",
    for_kmeans_fullframe=None,
):
    VIDEO_NAME = video_name
    print(f"\033[32m{VIDEO_NAME}を処理しています。=>k-means\033[0m")
    SAVE_PATH = f"{save_path}/k-means_temp/"
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        print("create SAVE PATH")
    else:
        try:
            os.makedirs(SAVE_PATH)
        except OSError as e:
            print(f"Failed to create directory {SAVE_PATH}. Reason: {e}")
    print(f"\033[32m一時作業フォルダーを作成しました。\033[0m=>{SAVE_PATH}")

    npy_image_list = create_npy_image_list(for_kmeans_array, kmeans_cnn)
    save_image_array = (
        for_kmeans_fullframe if for_kmeans_fullframe is not None else for_kmeans_array
    )

    npy_image_list_df = pd.DataFrame(npy_image_list)

    print(npy_image_list_df.shape)

    print(f"\033[32m{reduction_method.upper()} start\033[0m")
    tsne_results = build_embedding(
        npy_image_list_df, method=reduction_method, n_components=3
    )
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2", "TSNE3"])

    print("\033[32mK-means start\033[0m")
    kmeans = build_kmeans(tsne_df, cluster_num)
    print("\033[32mK-meansモデル構築完了\033[0m")

    tsne_df["label"] = kmeans.labels_
    tsne_df["tsne_idx"] = tsne_df.index
    tsne_idx_list = tsne_df.index.tolist()

    make_cluster_dir_parallel(
        save_image_array,
        SAVE_PATH,
        kmeans,
        format_flag,
        video_name,
        for_kmeans_frame_no,
        tsne_idx_list,
    )

    SELECTED_DIR = f"{save_path}/selected_imgs/"
    if not os.path.exists(SELECTED_DIR):
        try:
            os.makedirs(SELECTED_DIR)
        except OSError as e:
            print(f"Failed to create directory {SELECTED_DIR}. Reason: {e}")

    cluster_centers = kmeans.cluster_centers_
    selected_images = []

    for i in range(cluster_num):
        cluster_images = tsne_df[tsne_df["label"] == i]

        # print(cluster_images.head(4))

        cluster_center = cluster_centers[i]
        distances = np.linalg.norm(
            cluster_images[["TSNE1", "TSNE2", "TSNE3"]].values - cluster_center, axis=1
        )

        closest_image_idx = distances.argmin()
        closest_tsne_idx = int(cluster_images.iloc[closest_image_idx]["tsne_idx"])

        wildcard_part = "******"
        selected_image_path = f"{SAVE_PATH}cluster{i}/class{i}_{video_name}-{wildcard_part}_idx{closest_tsne_idx}.{format_flag}"

        # print(f"Selected image path: {selected_image_path}")  # デバッグ用出力

        selected_images.append(selected_image_path)

    # ワイルドカードを展開してコピー
    for selected_image_path in selected_images:
        matched_files = glob.glob(selected_image_path)
        if matched_files:
            for matched_file in matched_files:
                # print(f"Copying image from {matched_file} to {SELECTED_DIR}")  # デバッグ用出力
                shutil.copy(matched_file, SELECTED_DIR)
        else:
            print(f"No files matched for {selected_image_path}")

    print("\033[32mファイル出力完了\033[0m")

    # dev_flagで散布図の保存とクラスタリングされた全身画像の保存を切り替える
    if dev_flag:
        axis_label_prefix = "UMAP" if reduction_method.lower() == "umap" else "TSNE"
        reduction_name = reduction_method.lower()

        d2_fig, d2_ax = plot_tsne2d(tsne_df, axis_label_prefix=axis_label_prefix)
        graph_savepath = f"{save_path}/{reduction_name}2d.png"
        d2_fig.savefig(graph_savepath)

        d3_fig, d3_ax = plot_tsne3d(tsne_df, axis_label_prefix=axis_label_prefix)

        frames = 360
        with tqdm(total=frames) as pbar:
            ani = FuncAnimation(
                d3_fig, update, frames=frames, fargs=(d3_ax, pbar), interval=50
            )

            writer = PillowWriter(fps=30)
            animation_savepath = f"{save_path}/{reduction_name}3d_animation.gif"
            ani.save(animation_savepath, writer=writer)
    else:
        try:
            shutil.rmtree(SAVE_PATH)
        except Exception as e:
            print(f"Failed to delete {SAVE_PATH}. Reason: {e}")
        print("\033[32m一時ファイルを削除しました\033[0m")

    # os.system(f"open {SELECTED_DIR}")


if __name__ == "__main__":
    pass
