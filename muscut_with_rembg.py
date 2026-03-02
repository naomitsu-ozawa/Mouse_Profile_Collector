import argparse
import os
import platform

from muscut_tools import muscut_extract_ok_frames, webcam_list, kmeans_ok_frames

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os_name = platform.system()


def main(
    movie_path,
    device_flag,
    image_flag,
    tool,
    show_flag,
    cluster_num,
    mode,
    wc_flag,
    camera_list,
    all_extract,
    cnn_conf,
    pint,
    dev_flag,
):
    print("\033[32mmuscut.py_start\033[0m")
    from ultralytics import YOLO

    if camera_list is True:
        mode = "camera_list"
        print("webcamera list")

        webcam_list.webcam_list()
        return
    if movie_path is None:
        print("-fオプションでファイルのパスかWebcamを指定して下さい。")
        return

    if image_flag is None:
        image_flag = "png"

    if tool is None:
        tool = "extract_ok_frames"

    if mode is None:
        if os_name == "Darwin":
            mode = "coreml"
        elif os_name == "Linux" or os_name == "Windows":
            mode = "tf_pt"

    if mode == "coreml":
        import coremltools as ct

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        yolo_model = YOLO("muscut_models/yolo.mlmodel", task="detect")
        cnn_model = ct.models.MLModel("muscut_models/ct_cnn.mlmodel")

        # k-means test
        kmeans_cnn = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet", alpha=1.0
        )

    elif mode == "tf_pt":

        import logging
        import warnings

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)
        tf.get_logger().setLevel("INFO")
        tf.autograph.set_verbosity(0)
        tf.get_logger().setLevel(logging.ERROR)

        # GPU_flag = tf.test.is_gpu_available()
        GPU_flag = tf.config.list_physical_devices("GPU")

        if device_flag is None or device_flag == "":
            if GPU_flag:
                if os_name == "Darwin":
                    device_flag = "mps"
                    print("GPU: metal")
                else:
                    device_flag = "cuda"
                    print("GPU: CUDA")
            else:
                device_flag = "cpu"
                print("CPU")
        else:
            if device_flag == "cpu":
                print("CPU")
            elif device_flag == "mps":
                print("GPU: metal")
            elif device_flag == "cuda":
                print("GPU: CUDA")

        yolo_model = YOLO("muscut_models/yolo.pt")
        cnn_model = tf.keras.models.load_model("muscut_models/cnn/savedmodel")

        # k-means test
        kmeans_cnn = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet", alpha=1.0
        )

    if cnn_conf is None:
        cnn_conf = 0.7
        print(f"cnn_conf: default {cnn_conf}")
    else:
        print(f"cnn_conf = {cnn_conf}")

    if pint is None:
        pint = 2600
        print(f"pint_check_threshold: default 2600")
    else:
        print(f"pint_check_threshold = {pint}")

    if tool == "default":
        muscut_default.muscut(
            movie_path,
            device_flag,
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
        )

    elif tool == "extract_ok_frames":
        muscut_extract_ok_frames.main(
            movie_path,
            device_flag,
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
            dev_flag,
        )


def get_args():
    parser = argparse.ArgumentParser(
        prog="muscut_tool",  # プログラム名
        # usage="",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    # opiton file name
    parser.add_argument(
        "-f",
        "--movie_path",
        help="ファイルのパスかwebcamを指定して下さい。['file_path','movie_path','webcam']",
        # required=True,
    )

    # option mode
    parser.add_argument(
        "-m",
        "--mode",
        help="tensorflow&pytorchを利用する or coremlを利用する['tf_pt','coreml']",
        type=str,
    )

    # option tf&pt device option
    parser.add_argument(
        "-d",
        "--device_flag",
        help="mode=tf_ptで利用するデバイスを指定して下さい。指定しない場合は、['cpu']で実行します。['cpu','cuda']",
        type=str,
    )

    # option tools
    parser.add_argument(
        "-t",
        "--tool",
        help="使用するツールを指定して下さい。['default','kmeans_image_extractor']",
        type=str,
    )

    # option image format
    parser.add_argument(
        "-i",
        "--image_format",
        help="画像の保存フォーマットを指定して下さい。指定しない場合は、['png']で保存します。['jpg','png']",
    )

    # option preview show
    parser.add_argument(
        "-s",
        "--show_flag",
        action="store_true",
        help="検知状況を表示します[True or False]",
    )

    parser.add_argument(
        "-c", "--cnn_conf", type=float, help="画像分類モデルの閾値を、少数で設定"
    )

    # option preview show
    parser.add_argument(
        "-wc",
        "--without_cnn",
        action="store_true",
        help="CNNモデルによる分類をスキップします。",
    )

    # option extract number
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="抽出枚数",
    )

    # option pint check
    parser.add_argument(
        "-p",
        "--pint",
        type=int,
        help="ピントチェックの閾値、デフォルト2600",
    )

    # option webcamera list
    parser.add_argument(
        "-cl",
        "--camera_list",
        action="store_true",
        help="webcamera list",
    )

    # option all extract
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="all_extract",
    )

    # option dev_flag
    parser.add_argument(
        "-dev",
        "--dev_flag",
        action="store_true",
        default=False,
        help="dev_flag",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":
    args = get_args()

    movie_path = args.movie_path
    device_name = args.device_flag
    image_format = args.image_format
    tool = args.tool
    show_flag = args.show_flag
    cluster_num = args.number
    pint = args.pint
    mode = args.mode
    wc_flag = args.without_cnn
    camera_list = args.camera_list
    all_extract = args.all
    cnn_conf = args.cnn_conf
    dev_flag = args.dev_flag

    main(
        movie_path,
        device_name,
        image_format,
        tool,
        show_flag,
        cluster_num,
        mode,
        wc_flag,
        camera_list,
        all_extract,
        cnn_conf,
        pint,
        dev_flag,
    )
    print("\033[32m処理が完了しました。\n\033[0m")
