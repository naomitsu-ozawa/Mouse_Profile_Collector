import os
from pathlib import Path

import cv2
from tqdm import tqdm

from muscut_tools import kmeans


def main(movie_path, format_flag, cluster_num, kmeans_cnn):
    if cluster_num is None:
        cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
    else:
        pass
    # source video path
    movie_file = movie_path
    movie_file_name = Path(movie_file).stem
    # 保存先
    save_path = f"video_to_image/kmeans/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    for_kmeans_array = []
    for_kmeans_frame_no = []

    cv2movie = cv2.VideoCapture(movie_file)
    nframe = int(cv2movie.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = range(nframe)

    for idx, frame in tqdm(enumerate(frames)):
        ret, frame = cv2movie.read()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        # img = cv2.resize(img, (1280, 720))
        for_kmeans_array.append(img)
        for_kmeans_frame_no.append(idx)

    kmeans.kmeans_main(
        save_path,
        movie_file_name,
        for_kmeans_array,
        cluster_num,
        format_flag,
        for_kmeans_frame_no,
        kmeans_cnn,
        dev_flag=False,
    )
    print("\033[32mAll Done!\033[0m")
