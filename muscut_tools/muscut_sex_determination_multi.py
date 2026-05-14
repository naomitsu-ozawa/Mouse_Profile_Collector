# %%
import datetime
import math
import os
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# %%
class StrRe(str):
    def __init__(self, var):
        self.var = var
        pass

    def __eq__(self, pattern):
        return True if re.search(pattern, self.var) is not None else False


def muscut(
    movie_path,
    device,
    image_flag,
    show_flag,
    yolo_model,
    cnn_model,
    mode,
    cluster_num,
    wc_flag,
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
    elif mode == "tf":
        import tensorflow as tf

        running_mode = "TensorFlow&PyTorch"

    os_name = platform.system()

    print(f"OS:{os_name}")
    print(f"runing mode :{running_mode}")
    print(f"image format:[{image_flag}]です。")

    if os_name == "Darwin":
        cap = cv2.VideoCapture(movie_path, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(movie_path)

    if webcam_flag:
        total_frames = None
        print("use_webcam")
    else:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    count_male = 0
    count_female = 0
    n = 0

    if mode == "coreml":
        input_name = cnn_model.get_spec().description.input[0].name

    # Loop through the video frames
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            time.sleep(0.000005)
            pbar.update(1)
            n += 1
            # Read a frame from the video
            success, frame = cap.read()
            cnn_result = 0

            if success:
                # Run YOLOv8 inference on the frame
                results = yolo_model(frame, verbose=False)
                try:
                    if mode == "coreml":
                        result = results[0].numpy()
                    elif mode == "tf":
                        result = results[0].cpu().numpy()
                    else:
                        print("modeを指定して下さい")
                        return

                    ori_img = result.orig_img
                    length = result.boxes.shape[0]
                    box_count_male = 0
                    box_count_female = 0

                    for i in range(length):
                        box = result[i].boxes.xywh
                        cv_box = result[i].boxes.xyxy
                        # name = result.names
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

                        croped = ori_img[left_top_y:right_btm_y, left_top_x:right_btm_x]
                        croped = cv2.resize(croped, (224, 224))

                        if not wc_flag:
                            if mode == "coreml":
                                img_np = np.array(croped).astype(np.float32)
                                img_np = img_np[np.newaxis, :, :, :]
                                cnn_result = cnn_model.predict({input_name: img_np})
                                cnn_result_male = cnn_result["Identity"][0][0]
                                cnn_result_female = cnn_result["Identity"][0][1]
                                # cnn_result_male = round(float(cnn_result_male), 4)
                                # cnn_result_female = round(float(cnn_result_female), 4)

                                if cnn_result_male > cnn_result_female:
                                    count_male += 1
                                    box_count_male += 1
                                    cv2.rectangle(
                                        frame,
                                        (cv_top_x, cv_top_y),
                                        (cv_btm_x, cv_btm_y),
                                        (255, 0, 0),
                                        thickness=3,
                                        lineType=cv2.LINE_AA,
                                    )
                                    cv2.rectangle(
                                        frame,
                                        (cv_top_x, cv_top_y),
                                        (cv_top_x + 250, cv_top_y + 40),
                                        (255, 0, 0),
                                        thickness=-1,
                                        lineType=cv2.LINE_AA,
                                    )
                                    cv2.putText(
                                        frame,
                                        text="male",
                                        org=(cv_top_x, cv_top_y + 20),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0,
                                        color=(250, 250, 250),
                                        thickness=2,
                                    )

                                elif cnn_result_female > cnn_result_male:
                                    count_female += 1
                                    box_count_female += 1
                                    cv2.rectangle(
                                        frame,
                                        (cv_top_x, cv_top_y),
                                        (cv_btm_x, cv_btm_y),
                                        (0, 0, 255),
                                        thickness=3,
                                        lineType=cv2.LINE_AA,
                                    )
                                    cv2.rectangle(
                                        frame,
                                        (cv_top_x, cv_top_y),
                                        (cv_top_x + 250, cv_top_y + 40),
                                        (0, 0, 255),
                                        thickness=-1,
                                        lineType=cv2.LINE_AA,
                                    )
                                    cv2.putText(
                                        frame,
                                        text="female",
                                        org=(cv_top_x, cv_top_y + 20),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0,
                                        color=(250, 250, 255),
                                        thickness=2,
                                    )

                        #     elif mode == "tf":
                        #         data = np.array(croped).astype(np.float32)
                        #         data = data[tf.newaxis]
                        #         x = tf.keras.applications.mobilenet_v3.preprocess_input(
                        #             data
                        #         )
                        #         cnn_result = cnn_model(x, training=False)
                        #         cnn_result = cnn_result.numpy()
                        #         cnn_result = cnn_result[0]
                        #         if cnn_result[1] > 0.8:
                        #             for_kmeans_array.append(croped)
                        #             count += 1
                        # else:
                        #     # cnn_result = "without cnn"
                        #     for_kmeans_array.append(croped)
                        #     count += 1

                except (IndexError, cv2.error):
                    cnn_result = 0
                    pass

                if wc_flag:
                    cnn_result = "without cnn"

                if show_flag is True:
                    # Visualize the results on the frame
                    # annotated_frame = results[0].plot(line_width=(3))
                    annotated_frame = frame
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.rectangle(annotated_frame, (0, 0), (300, 170), (80, 80, 80), -1)

                    text_1_1 = f"Male___:{box_count_male}"
                    text_1_2 = f"Female_:{box_count_female}"

                    # text_2_1 = f"Extractable images:{count_male}"
                    # text_2_2 = f"Extractable images:{count_female}"

                    cv2.putText(
                        annotated_frame,
                        text_1_1,
                        (5, 27),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(250, 250, 250),
                    )

                    cv2.putText(
                        annotated_frame,
                        text_1_2,
                        (5, 67),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.0,
                        color=(250, 250, 250),
                    )

                    if not webcam_flag:
                        prog = round(n / total_frames * 100)
                        prog_bar = round(prog * 2.54)
                        text_3 = f"|{n}/{round(total_frames)}|{prog}%|"
                        cv2.rectangle(
                            annotated_frame,
                            (5, 105),
                            (prog_bar, 120),
                            (250, 250, 250),
                            -1,
                        )
                    else:
                        text_3 = "webcam"

                    cv2.putText(
                        annotated_frame,
                        text_3,
                        (5, 97),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(250, 250, 250),
                    )

                    # Display the annotated frame
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
    # cv2.destroyWindow("YOLOv8 Inference")

    print("\033[32m顔検出完了\033[0m")

    print(f"検出数♂：[{count_male}]")
    print(f"検出数♀：[{count_female}]")

    m_count = count_male / total_frames
    f_count = count_female / total_frames

    print(f"male:{m_count}/female:{f_count}")

    # if cluster_num is None:
    #     cluster_num = int(input("\033[32m抽出する枚数を入力してください\033[0m >"))
    # else:
    #     pass

    # save_path = f"croped_image/{movie_file_name}"
    # os.makedirs(save_path, exist_ok=True)

    # kmeans.kmeans_main(
    #     save_path, movie_file_name, for_kmeans_array, cluster_num, image_flag
    # )
    print("\033[32mAll Done!\033[0m")
