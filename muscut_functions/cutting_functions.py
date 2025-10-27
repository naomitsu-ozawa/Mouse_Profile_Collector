from muscut_functions import cv_functions, global_functions
import numpy
import cv2
from tqdm import tqdm


def cutting(img, device, yolo_model, mode, output_folder, file_name):

    inf_image = cv_functions.black_back(img)

    yolo_conf = 0.5
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

    save_frame = img

    # length = result.boxes.shape[0]
    # for i in range(length):
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
    ) = cv_functions.crop_modified_xy(result[0])

    # try:
    #     cv_functions.check_coordinates(
    #         left_top_x,
    #         left_top_y,
    #         right_btm_x,
    #         right_btm_y,
    #         cv_top_x,
    #         cv_top_y,
    #         cv_btm_x,
    #         cv_btm_y,
    #     )
    # except:
    #     codinates = f"left_top_x:{left_top_x},left_top_y:{left_top_y},right_btm_x:{right_btm_x},right_btm_y:{right_btm_y},"

    #     error_msg = f"{file_name}:({codinates})-Coordinates cannot be negative"
    #     raise ValueError(error_msg)

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

    output_path = f"{output_folder}/{file_name}_with_rembg.png"

    # print(output_path)

    return croped, output_path


def process_cutting(images, device, yolo_model, mode, output_folder, imgnames):
    croped_images = []
    output_paths = []
    images = tqdm(images)
    for image, file_name in zip(images, imgnames):
        try:
            croped, output_path = cutting(
                image, device, yolo_model, mode, output_folder, file_name
            )
            croped_images.append(croped)
            output_paths.append(output_path)
        except:
            pass
    return croped_images, output_paths
