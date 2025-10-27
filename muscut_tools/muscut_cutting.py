from muscut_functions import cv_functions
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm


def black_back(img):
    # 画像の高さと幅を取得
    height, width = img.shape[:2]

    # 黒背景の画像を作成
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    # 元の画像のアルファチャンネルを無視して黒背景の画像に貼り付け
    black_background[:, :] = img[:, :, :3]

    return black_background


def cutting(img, device, yolo_model, mode, output_folder):
    file_name = os.path.basename(img)
    # Run YOLOv8 inference on the frame
    image = cv2.imread(img, -1)
    inf_image = black_back(image)
    # inf_image = cv2.imread(img)

    yolo_conf = 0.9
    if mode == "coreml":
        results = yolo_model(
            inf_image,
            max_det=1,  # max detecxtion num.
            conf=yolo_conf,  # object confidence threshold for detection
            verbose=False,
        )
    elif mode == "tf_pt":
        results = yolo_model(
            inf_image,
            max_det=1,  # max detecxtion num.
            device=device,
            conf=yolo_conf,  # object confidence threshold for detection
            verbose=False,
        )

    if mode == "coreml":
        result = results[0].numpy()
    elif mode == "tf_pt":
        result = results[0].cpu().numpy()

    save_frame = image

    length = result.boxes.shape[0]
    for i in range(length):
        # xy convert to square
        (
            left_top_x,
            left_top_y,
            right_btm_x,
            right_btm_y,
            cv_top_x,
            cv_top_y,
            cv_btm_x,
            cv_btm_y,
        ) = cv_functions.crop_modified_xy(result[i])

        checkk_result = cv_functions.check_coordinates(
            left_top_x,
            left_top_y,
            right_btm_x,
            right_btm_y,
            cv_top_x,
            cv_top_y,
            cv_btm_x,
            cv_btm_y,
        )

        if checkk_result == False:
            croped = cv_functions.crop_square_with_fill(
                save_frame,
                (cv_top_x + cv_btm_x) // 2,
                (cv_top_y + cv_btm_y) // 2,
                right_btm_x - left_top_x,
                right_btm_y - left_top_y,
            )
        elif checkk_result == True:
            croped = save_frame[left_top_y:right_btm_y, left_top_x:right_btm_x]

        croped = cv2.resize(croped, (224, 224))
        output_path = f"{output_folder}/{file_name}_{i}_with_rembg.png"
        # print(output_path)
        cv2.imwrite(output_path, croped)


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(input_path, device, yolo_model, mode):
    input_path = f"{input_path}/rembg_imgs"
    imgs = glob.glob(f"{input_path}/*.png")
    print("切り取り中・・・")
    imgs = tqdm(imgs)
    for img in imgs:
        output_folder = f"{os.path.dirname(os.path.dirname(img))}/with_rembg"
        # print(img)
        # print(output_folder)
        create_directory_if_not_exists(output_folder)
        cutting(img, device, yolo_model, mode, output_folder)
