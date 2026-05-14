import glob
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


def crop_modified_xy(result):
    box = result.boxes.xywh
    cv_box = result.boxes.xyxy
    xcenter = box[0][0]
    ycenter = box[0][1]
    width = box[0][2]
    height = box[0][3]

    cv_top_x = int(cv_box[0][0]) + 1
    cv_top_y = int(cv_box[0][1]) + 1
    cv_btm_x = int(cv_box[0][2]) - 1
    cv_btm_y = int(cv_box[0][3]) - 1

    if width > height:
        height = width
    elif height > width:
        width = height
    left_top_x = math.floor(xcenter - (width / 2))
    left_top_y = math.floor(ycenter - (height / 2))
    right_btm_x = math.floor(xcenter + (width / 2))
    right_btm_y = math.floor(ycenter + (height / 2))

    return (
        left_top_x,
        left_top_y,
        right_btm_x,
        right_btm_y,
        cv_top_x,
        cv_top_y,
        cv_btm_x,
        cv_btm_y,
    )


def pint_check(pred_croped, pint):
    p = cv2.Sobel(
        pred_croped, dx=1, dy=1, ddepth=cv2.CV_8U, ksize=5, scale=1, delta=50
    ).var()
    if p > pint:
        pint_check = True
    else:
        pint_check = False

    return pint_check


def display_detected_frame(
    ori_img, txt, cv_top_x, cv_top_y, cv_btm_x, cv_btm_y, frame_color
):
    cv2.rectangle(
        ori_img,
        (cv_top_x, cv_top_y),
        (cv_btm_x, cv_btm_y),
        frame_color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        ori_img,
        (cv_top_x, cv_top_y),
        (cv_top_x + 250, cv_top_y + 40),
        frame_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        ori_img,
        text=txt,
        org=(cv_top_x, cv_top_y + 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(250, 250, 250),
        thickness=2,
    )


def display_preview_screen(
    annotated_frame, cnn_bar, count, total_frames, pip_croped, webcam_flag, n
):
    annotated_frame = cv2.resize(annotated_frame, (1280, 720))

    # picture in picture coordinates
    cv2.rectangle(annotated_frame, (0, 0), (256, 320), (80, 80, 80), -1)

    text_1 = "CNN_score:"
    text_2 = f"Extractable images:{count}"

    cv2.putText(
        annotated_frame,
        text_1,
        (5, 17),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.5,
        color=(250, 250, 250),
    )
    cv2.putText(
        annotated_frame,
        text_2,
        (5, 37),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.5,
        color=(250, 250, 250),
    )
    cv2.rectangle(
        annotated_frame,
        (100, 5),
        (cnn_bar, 20),
        (250, 250, 250),
        -1,
    )

    if not webcam_flag:
        prog = round(n / total_frames * 100)
        prog_bar = round(prog * 2.54)
        text_3 = f"|{n}/{round(total_frames)}|{prog}%|"
        cv2.rectangle(
            annotated_frame,
            (5, 65),
            (prog_bar, 70),
            (250, 250, 250),
            -1,
        )
    else:
        text_3 = "webcam"

    cv2.putText(
        annotated_frame,
        text_3,
        (5, 57),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=0.5,
        color=(250, 250, 250),
    )
    # PiP
    pip_x = 16
    pip_y = 80
    pip_h, pip_w = pip_croped.shape[:2]
    if pip_croped.shape == (224, 224, 3):
        annotated_frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_croped

    return annotated_frame


def read_image(img_path):
    imgname = os.path.basename(img_path)
    return cv2.imread(img_path, -1), imgname


def read_images_parallel(imgs):
    # プロセス数を取得（最大はCPUコア数）
    num_processes = os.cpu_count()

    # プールを作成
    pool = Pool(processes=num_processes)

    with tqdm(total=len(imgs), desc="Loading...") as pbar:
        # pool.imap() からの値を1つずつ取得して処理する
        rimgs = []
        imgnames = []
        for result in pool.imap(read_image, imgs):
            img, img_name = result
            rimgs.append(img)
            imgnames.append(img_name)
            pbar.update(1)

    # プールを閉じる
    pool.close()
    pool.join()

    print("All images loaded.")
    return rimgs, imgnames


def black_back(img):
    # 画像の高さと幅を取得
    height, width = img.shape[:2]

    # 黒背景の画像を作成
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    # 元の画像のアルファチャンネルを無視して黒背景の画像に貼り付け
    black_background[:, :] = img[:, :, :3]

    return black_background


def safe_close_all_windows():
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        # Headless OpenCV builds do not provide HighGUI window teardown.
        pass


def check_coordinates(
    left_top_x,
    left_top_y,
    right_btm_x,
    right_btm_y,
    cv_top_x,
    cv_top_y,
    cv_btm_x,
    cv_btm_y,
):
    if (
        left_top_x < 0
        or left_top_y < 0
        or right_btm_x < 0
        or right_btm_y < 0
        or cv_top_x < 0
        or cv_top_y < 0
        or cv_btm_x < 0
        or cv_btm_y < 0
    ):
        # raise ValueError("Coordinates cannot be negative")
        return False
    else:
        return True


def crop_square_with_fill(img, xcenter, ycenter, width, height):
    """
    YOLO出力座標から正方形クロップを行い、
    画像外にはみ出した部分を黒で塗りつぶして補完。
    """

    H, W = img.shape[:2]
    side = int(max(width, height))  # 長辺を基準に正方形化

    # 元の正方形座標
    left_top_x = math.floor(xcenter - side / 2)
    left_top_y = math.floor(ycenter - side / 2)
    right_btm_x = left_top_x + side
    right_btm_y = left_top_y + side

    # パディング計算
    pad_left = max(0, -left_top_x)
    pad_top = max(0, -left_top_y)
    pad_right = max(0, right_btm_x - W)
    pad_bottom = max(0, right_btm_y - H)

    # 画像内クロップ座標を修正
    x1 = max(0, left_top_x)
    y1 = max(0, left_top_y)
    x2 = min(W, right_btm_x)
    y2 = min(H, right_btm_y)

    # 実際のクロップ
    cropped = img[y1:y2, x1:x2]

    # 黒塗りパディング（ここが「塗りつぶし」部分！）
    filled = cv2.copyMakeBorder(
        cropped,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # 黒
    )

    return filled
