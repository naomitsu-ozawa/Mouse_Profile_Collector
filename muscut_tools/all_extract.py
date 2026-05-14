# %%
import math
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


# %%
def read_frames(movie_path, queue, device_flag, show_flag):
    os_name = platform.system()

    if os_name == "Darwin":
        model = YOLO("moucut_models/yolo.mlmodel", task="detect")
        cap = cv2.VideoCapture(movie_path, cv2.CAP_AVFOUNDATION)
        device_name = ""
    else:
        model = YOLO("moucut_models/b6.pt")
        device_name = device_flag
        cap = cv2.VideoCapture(movie_path)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Loop through the video frames
    detection_count = 0
    n = 0
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            time.sleep(0.000005)
            pbar.update(1)
            n += 1
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame

                if os_name == "Darwin":
                    results = model(frame, verbose=False)
                else:
                    results = model(frame, device=device_name, verbose=False)

                try:
                    if os_name == "Darwin":
                        result = results[0].numpy()
                    else:
                        result = results[0].cpu().numpy()

                    ori_img = result.orig_img
                    length = result.boxes.shape[0]
                    for i in range(length):
                        box = result[i].boxes.xywh
                        # name = result.names
                        xcenter = box[0][0]
                        ycenter = box[0][1]
                        width = box[0][2]
                        height = box[0][3]

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
                        detection_count += 1
                        queue.put(croped)

                except (IndexError, cv2.error):
                    pass

                # Display the annotated frame
                if show_flag is True:
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot(line_width=(3))
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.rectangle(
                        annotated_frame, (0, 0), (50 + 800, 50 + 100), (80, 80, 80), -1
                    )
                    text_1 = f"Number of saved images:{detection_count}"
                    text_2 = "EXIT => 'q'key"

                    cv2.putText(
                        annotated_frame,
                        text_1,
                        (50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )
                    cv2.putText(
                        annotated_frame,
                        text_2,
                        (50, 100),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(250, 250, 250),
                    )
                    cv2.imshow("Inference", annotated_frame)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            else:
                # Break the loop if the end of the video is reached
                break
    # Release the video capture object and close the display window
    cap.release()
    cv_functions.safe_close_all_windows()
    # cv2.destroyWindow("YOLOv8 Inference")


def write_frames(queue, save_path, image_flag):
    count = 0
    while True:
        frame = queue.get()
        if frame is None:
            break
        count_number = str(count).zfill(6)
        if image_flag == "jpg":
            cv2.imwrite(save_path + "/extract_{}.jpg".format(count_number), frame)
        else:
            cv2.imwrite(save_path + "/extract_{}.png".format(count_number), frame)
        count += 1


def moucut(movie_path, decive_flag, image_flag, show_flag):
    print("runing_All_Extract_version")
    print(f"物体検出に{decive_flag}を利用します。")
    print(f"画像の保存形式は[{image_flag}]です。")

    if movie_path == "webcam":
        movie_path = 0
        movie_file_name = "webcam"
    else:
        movie_path = movie_path
        movie_file_name = Path(movie_path).stem

    save_path = f"croped_image/all_extract_image/{movie_file_name}"
    os.makedirs(save_path, exist_ok=True)

    queue = mp.Queue(maxsize=10)
    reader = mp.Process(
        target=read_frames, args=(movie_path, queue, decive_flag, show_flag)
    )
    writer = mp.Process(target=write_frames, args=(queue, save_path, image_flag))

    reader.start()
    writer.start()
    reader.join()
    queue.put(None)
    writer.join()
    print("\033[32m顔検出終了\033[0m")

    # print(f"検出数：[{count}]")
