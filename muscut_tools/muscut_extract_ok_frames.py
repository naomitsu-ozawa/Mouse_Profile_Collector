# %%
import datetime
import math
import os
import platform
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from muscut_functions import cv_functions
from muscut_tools import (
    all_save,
    kmeans,
    kmeans_ok_frames,
    muscut_cutting,
    muscut_rembg,
    muscut_cutting_multi_process,
    muscut_rembg_multi_process,
)


# %%
def save_images_to_directory(images, file_names, directory_path):
    os.makedirs(directory_path, exist_ok=True)
    for image, file_name in zip(images, file_names):
        cv2.imwrite(str(Path(directory_path) / file_name), image)


class StrRe(str):
    def __init__(self, var):
        self.var = var
        pass

    def __eq__(self, pattern):
        return True if re.search(pattern, self.var) is not None else False


def main(
    movie_path,
    device,
    image_flag,
    show_flag,
    yolo_model,
    cnn_model,
    mode,
    cluster_num,
    wc_flag,
    all_extract,
    cnn_conf,
    pint,
    kmeans_cnn,
    dev_flag
):
    match StrRe(movie_path):
        case "webcam*":
            timezone = datetime.timezone(datetime.timedelta(hours=+9), "JST")
            dt = datetime.datetime.now()
            dt = dt.astimezone(timezone)
            dt = "{0:%Y-%m-%d-%H-%M-%S}".format(dt)
            movie_path = int(movie_path[-1])
            movie_file_name = f"webcam{dt}"
            webcam_flag = True
        case _:
            movie_file_name = Path(movie_path).stem
            webcam_flag = False

    if mode == "coreml":
        running_mode = "CoreML"
    elif mode == "tf_pt":

        import logging
        import warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        running_mode = "TensorFlow&PyTorch"

    os_name = platform.system()

    print(f"OS:{os_name}")
    print(f"runing mode :{running_mode}")
    print(f"image format:[{image_flag}]です。")

    if os_name == "Darwin":
        cap = cv2.VideoCapture(movie_path, cv2.CAP_AVFOUNDATION)
    else:
        # cap = cv2.VideoCapture(movie_path, cv2.CAP_ANY)
        cap = cv2.VideoCapture(movie_path, cv2.CAP_FFMPEG)

    if webcam_flag:
        total_frames = None
        print("use_webcam")
    else:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for_kmeans_array = []
    for_kmeans_fullframe = []
    for_kmeans_frame_no = []

    count = 0
    n = 0

    if mode == "coreml":
        input_name = cnn_model.get_spec().description.input[0].name
    elif mode == "tf_pt":
        signature_keys = list(cnn_model.signatures.keys())
        infer = cnn_model.signatures[signature_keys[0]]
        outputs = list(infer.structured_outputs.keys())[0]
        # print("trt_preprossece_ok")

    pip_croped = np.zeros((224, 224, 3))

    # Loop through the video frames
    print(f"\033[32m{movie_file_name}:Processing...\033[0m")
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            time.sleep(0.000005)
            pbar.update(1)
            n += 1
            frame_no = n - 1
            # Read a frame from the video
            success, frame = cap.read()

            cnn_result = 0
            cnn_bar = 101

            # yolo conf setting
            yolo_conf = 0.5

            if success:
                # test 4K重いのでリサイズする？
                # frame = cv2.resize(frame, (1920, 1080))

                # Run YOLOv8 inference on the frame
                if mode == "coreml":
                    results = yolo_model(
                        frame,
                        # max_det=1,  # max detecxtion num.
                        conf=yolo_conf,  # object confidence threshold for detection
                        verbose=False,
                    )
                elif mode == "tf_pt":
                    results = yolo_model(
                        frame,
                        # max_det=1,  # max detecxtion num.
                        device=device,
                        conf=yolo_conf,  # object confidence threshold for detection
                        verbose=False,
                    )

                try:
                    if mode == "coreml":
                        result = results[0].numpy()
                    elif mode == "tf_pt":
                        result = results[0].cpu().numpy()
                    else:
                        print("modeを指定して下さい")
                        return

                    ori_img = frame
                    fullframe = frame.copy()
                    save_frame = frame.copy()

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

                        croped = save_frame[
                            left_top_y:right_btm_y, left_top_x:right_btm_x
                        ]

                        croped = cv2.resize(croped, (224, 224))
                        pred_croped = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)

                        # pint check
                        pint_check = cv_functions.pint_check(pred_croped, pint)

                        if not wc_flag:
                            if mode == "coreml":
                                if pint_check:
                                    img_np = np.array(pred_croped).astype(np.float32)
                                    img_np = img_np[np.newaxis, :, :, :]
                                    cnn_result = cnn_model.predict({input_name: img_np})
                                    cnn_result = cnn_result["Identity"][0][1]
                                else:
                                    cnn_result = 0

                            elif mode == "tf_pt":
                                if pint_check:
                                    data = np.array(pred_croped).astype(np.float32)
                                    data = data[tf.newaxis]
                                    x = tf.keras.applications.mobilenet_v3.preprocess_input(
                                        data
                                    )
                                    x = tf.constant(x)
                                    cnn_result = infer(x)
                                    cnn_result = cnn_result[outputs].numpy()
                                    cnn_result = cnn_result[0][1]
                                else:
                                    cnn_result = 0
                        else:
                            cnn_result = 1.0

                        cnn_result = round(float(cnn_result), 4)
                        cnn_bar = int(cnn_result * 139 + 101)

                        if cnn_result > cnn_conf:
                            for_kmeans_array.append(croped)
                            for_kmeans_fullframe.append(fullframe)
                            for_kmeans_frame_no.append(str(frame_no))

                            count += 1
                            pip_croped = croped

                            cv_functions.display_detected_frame(
                                ori_img,
                                "OK",
                                cv_top_x,
                                cv_top_y,
                                cv_btm_x,
                                cv_btm_y,
                                (250, 0, 0),
                            )
                        else:
                            cv_functions.display_detected_frame(
                                ori_img,
                                "Not Detect",
                                cv_top_x,
                                cv_top_y,
                                cv_btm_x,
                                cv_btm_y,
                                (127, 127, 127),
                            )

                except (IndexError, cv2.error):
                    cnn_result = 0
                    pass

                if wc_flag:
                    cnn_result = "without cnn"

                if show_flag is True:
                    # Visualize the results on the frame
                    if wc_flag is True:
                        annotated_frame = results[0].plot(line_width=(3))
                    else:
                        annotated_frame = ori_img
                    annotated_frame = cv_functions.display_preview_screen(
                        annotated_frame,
                        cnn_bar,
                        count,
                        total_frames,
                        pip_croped,
                        webcam_flag,
                        n,
                    )

                    cv2.imshow("Inference", annotated_frame)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    cap.release()
    cv_functions.safe_close_all_windows()

    print("\033[32m顔検出完了\033[0m")

    print(f"検出数：[{count}]")

    save_path = f"croped_image/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    if all_extract is True:
        all_save.main(movie_path, for_kmeans_fullframe, for_kmeans_frame_no,  image_flag)
    else:
        if cluster_num is None:
            cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
        else:
            pass
        if count < cluster_num:
            cluster_num = count - 1
        else:
            pass

        kmeans_ok_frames.kmeans_main(
            save_path,
            movie_file_name,
            for_kmeans_array,
            for_kmeans_fullframe,
            cluster_num,
            image_flag,
            for_kmeans_frame_no,
            kmeans_cnn,
            dev_flag
        )

    #####
    input_path = f"{save_path}"
    # rembg
    rembg_images, file_names = muscut_rembg_multi_process.main(input_path)

    if dev_flag:
        before_rembg_dir = f"{save_path}/before_rembg_full"
        after_rembg_dir = f"{save_path}/after_rembg_full"
        selected_dir = f"{save_path}/selected_imgs"

        if os.path.exists(before_rembg_dir):
            shutil.rmtree(before_rembg_dir)
        shutil.copytree(selected_dir, before_rembg_dir)
        save_images_to_directory(rembg_images, file_names, after_rembg_dir)

    # muscut cutting
    muscut_cutting_multi_process.main(input_path, rembg_images, file_names, device, yolo_model, mode)

    try:
        shutil.rmtree(f"{save_path}/selected_imgs")
    except Exception as e:
        print(f"Failed to delete. Reason: {e}")

    print(f"\033[32m{movie_file_name}: Done!\033[0m")
