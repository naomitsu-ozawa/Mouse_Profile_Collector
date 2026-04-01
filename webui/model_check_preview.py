import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path.cwd()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from muscut_functions import cv_functions

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def emit(message: str) -> None:
    print(message, flush=True)


def transcode_for_browser(source_path: Path, output_path: Path) -> None:
    temp_output = output_path.with_name(f"{output_path.stem}-browser{output_path.suffix}")
    if temp_output.exists():
        temp_output.unlink()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temp_output.exists():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "ブラウザ再生用動画への変換に失敗しました。")

    source_path.unlink(missing_ok=True)
    temp_output.replace(output_path)


def load_models():
    os_name = platform.system()
    from ultralytics import YOLO

    if os_name == "Darwin":
        import coremltools as ct

        yolo_model = YOLO("muscut_models/yolo.mlmodel", task="detect")
        cnn_model = ct.models.MLModel("muscut_models/ct_cnn.mlmodel")
        return os_name, "coreml", yolo_model, cnn_model, None, None

    import tensorflow as tf

    warnings_logger = tf.get_logger()
    warnings_logger.setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

    yolo_model = YOLO("muscut_models/yolo.pt")
    cnn_model = tf.keras.models.load_model("muscut_models/cnn/savedmodel")
    signature_keys = list(cnn_model.signatures.keys())
    infer = cnn_model.signatures[signature_keys[0]]
    outputs = list(infer.structured_outputs.keys())[0]
    return os_name, "tf_pt", yolo_model, cnn_model, infer, outputs, device


def run_model_check(
    movie_path: Path,
    output_path: Path,
    yolo_conf: float = 0.5,
    cnn_conf: float = 0.7,
    pint: int = 2600,
) -> Path:
    loaded = load_models()
    if len(loaded) == 6:
        os_name, mode, yolo_model, cnn_model, infer, outputs = loaded
        device = None
    else:
        os_name, mode, yolo_model, cnn_model, infer, outputs, device = loaded

    if os_name == "Darwin":
        cap = cv2.VideoCapture(str(movie_path), cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(str(movie_path), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("確認用動画を開けませんでした。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_name(f"{output_path.stem}-raw{output_path.suffix}")
    raw_output_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (1280, 720))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("確認動画の出力ファイルを作成できませんでした。")

    if mode == "coreml":
        input_name = cnn_model.get_spec().description.input[0].name
        tf = None
    else:
        import tensorflow as tf

    emit("MODEL_CHECK_PHASE 動画を解析しています。")
    emit(f"MODEL_CHECK_FPS {fps:.2f}")

    frame_no = 0
    extracted_count = 0
    pip_croped = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_no += 1
            ori_img = frame.copy()
            save_frame = frame.copy()
            cnn_bar = 101

            if mode == "coreml":
                results = yolo_model(frame, conf=yolo_conf, verbose=False)
            else:
                results = yolo_model(frame, device=device, conf=yolo_conf, verbose=False)

            try:
                result = results[0].numpy() if mode == "coreml" else results[0].cpu().numpy()
                length = result.boxes.shape[0]
                for i in range(length):
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

                    check_result = cv_functions.check_coordinates(
                        left_top_x,
                        left_top_y,
                        right_btm_x,
                        right_btm_y,
                        cv_top_x,
                        cv_top_y,
                        cv_btm_x,
                        cv_btm_y,
                    )

                    if not check_result:
                        croped = cv_functions.crop_square_with_fill(
                            save_frame,
                            (cv_top_x + cv_btm_x) // 2,
                            (cv_top_y + cv_btm_y) // 2,
                            right_btm_x - left_top_x,
                            right_btm_y - left_top_y,
                        )
                    else:
                        croped = save_frame[left_top_y:right_btm_y, left_top_x:right_btm_x]

                    croped = cv2.resize(croped, (224, 224))
                    pred_croped = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)

                    pint_ok = cv_functions.pint_check(pred_croped, pint)
                    if mode == "coreml":
                        if pint_ok:
                            img_np = np.array(pred_croped).astype(np.float32)[np.newaxis, :, :, :]
                            cnn_result = cnn_model.predict({input_name: img_np})
                            cnn_result = cnn_result["Identity"][0][1]
                        else:
                            cnn_result = 0.0
                    else:
                        if pint_ok:
                            data = np.array(pred_croped).astype(np.float32)[tf.newaxis]
                            x = tf.keras.applications.mobilenet_v3.preprocess_input(data)
                            x = tf.constant(x)
                            cnn_result = infer(x)
                            cnn_result = cnn_result[outputs].numpy()[0][1]
                        else:
                            cnn_result = 0.0

                    cnn_result = round(float(cnn_result), 4)
                    cnn_bar = int(cnn_result * 139 + 101)

                    if cnn_result > cnn_conf:
                        extracted_count += 1
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
                pass

            annotated_frame = cv_functions.display_preview_screen(
                ori_img,
                cnn_bar,
                extracted_count,
                total_frames if total_frames > 0 else frame_no,
                pip_croped,
                False,
                frame_no,
            )
            writer.write(annotated_frame)

            total_for_progress = total_frames if total_frames > 0 else frame_no
            emit(f"MODEL_CHECK_PROGRESS {frame_no}/{total_for_progress}")

        emit("MODEL_CHECK_PHASE 確認動画を保存しています。")
    finally:
        cap.release()
        writer.release()

    if not raw_output_path.exists():
        raise RuntimeError("確認動画ファイルが生成されませんでした。")

    emit("MODEL_CHECK_PHASE ブラウザ再生用に動画を変換しています。")
    transcode_for_browser(raw_output_path, output_path)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model check preview video.")
    parser.add_argument("--movie", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--cnn-conf", type=float, default=0.7)
    parser.add_argument("--pint-threshold", type=int, default=2600)
    args = parser.parse_args()

    run_model_check(
        Path(args.movie),
        Path(args.output),
        yolo_conf=args.yolo_conf,
        cnn_conf=args.cnn_conf,
        pint=args.pint_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
