import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import platform

from flask import Flask, abort, after_this_request, jsonify, redirect, render_template, request, send_file, url_for


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SETTINGS_PATH = APP_DIR / "settings.json"
MUSCUT_MODELS_DIR = REPO_ROOT / "muscut_models"
CUSTOM_YOLO_DIR = MUSCUT_MODELS_DIR / "custom_yolo"
CUSTOM_CNN_DIR = MUSCUT_MODELS_DIR / "custom_cnn"
UPLOAD_DIR = APP_DIR / "uploads"
DOWNLOAD_DIR = APP_DIR / "downloads"
TMP_DIR = APP_DIR / "tmp"
WORKSPACE_DIR = APP_DIR / "workspaces"
MAX_TOTAL_CONCURRENT_JOBS = 8
MODE_CONCURRENCY_LIMITS = {
    "standard": 8,
    "rembg": 3,
}
SYSTEM_NAME = platform.system()
REMBG_MODEL_OPTIONS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]
DEFAULT_SETTINGS = {
    "yolo_model": "muscut_models/yolo.pt" if SYSTEM_NAME != "Darwin" else "muscut_models/yolo.mlmodel",
    "cnn_model": "muscut_models/cnn/savedmodel" if SYSTEM_NAME != "Darwin" else "muscut_models/ct_cnn.mlmodel",
    "yolo_conf": 0.5,
    "cnn_conf": 0.7,
    "rembg_model": "isnet-general-use",
    "rembg_alpha_matting": False,
    "rembg_alpha_matting_foreground_threshold": 240,
    "rembg_alpha_matting_background_threshold": 10,
    "rembg_alpha_matting_erode_size": 10,
}

for directory in (UPLOAD_DIR, DOWNLOAD_DIR, TMP_DIR, WORKSPACE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
for directory in (CUSTOM_YOLO_DIR, CUSTOM_CNN_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
SETTINGS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ACTIVE_JOB_IDS: set[str] = set()
JOB_QUEUE: deque[str] = deque()
PREVIEW_JOBS: dict[str, dict] = {}
PREVIEW_JOBS_LOCK = threading.Lock()
MODEL_CHECK_JOBS: dict[str, dict] = {}
MODEL_CHECK_JOBS_LOCK = threading.Lock()
PROGRESS_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
PROGRESS_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)")


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe or "upload.mp4"


def to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def list_model_options() -> dict[str, list[str]]:
    yolo_options: list[str] = [DEFAULT_SETTINGS["yolo_model"]]
    cnn_options: list[str] = [DEFAULT_SETTINGS["cnn_model"]]

    if SYSTEM_NAME == "Darwin":
        yolo_options.extend(
            sorted(
                {
                    to_repo_relative(path)
                    for path in CUSTOM_YOLO_DIR.rglob("*.mlmodel")
                    if path.is_file()
                }
            )
        )
        cnn_options.extend(
            sorted(
                {
                    to_repo_relative(path)
                    for path in CUSTOM_CNN_DIR.rglob("*.mlmodel")
                    if path.is_file()
                }
            )
        )
    else:
        yolo_options.extend(
            sorted(
                {
                    to_repo_relative(path)
                    for path in CUSTOM_YOLO_DIR.rglob("*.pt")
                    if path.is_file()
                }
            )
        )
        cnn_options.extend(
            sorted(
                {
                    to_repo_relative(saved_model_pb.parent)
                    for saved_model_pb in CUSTOM_CNN_DIR.rglob("saved_model.pb")
                }
            )
        )

    return {
        "yolo": yolo_options,
        "cnn": cnn_options,
    }


def build_choice_label(model_path: str, default_path: str, custom_root: str, model_kind: str) -> str:
    if model_path == default_path:
        return f"標準: {Path(model_path).name}"

    custom_prefix = f"{custom_root}/"
    if model_path.startswith(custom_prefix):
        suffix = model_path[len(custom_prefix):]
        return f"カスタム: {suffix}"

    return f"{model_kind}: {model_path}"


def build_model_choices() -> dict[str, list[dict[str, str]]]:
    options = list_model_options()
    return {
        "yolo": [
            {
                "value": option,
                "label": build_choice_label(option, DEFAULT_SETTINGS["yolo_model"], "muscut_models/custom_yolo", "YOLO"),
            }
            for option in options["yolo"]
        ],
        "cnn": [
            {
                "value": option,
                "label": build_choice_label(option, DEFAULT_SETTINGS["cnn_model"], "muscut_models/custom_cnn", "CNN"),
            }
            for option in options["cnn"]
        ],
    }


def save_settings(settings: dict[str, str | float | int | bool]) -> None:
    payload = json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
    temp_path = SETTINGS_PATH.with_name(f"{SETTINGS_PATH.name}.{uuid.uuid4().hex}.tmp")
    with SETTINGS_LOCK:
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(SETTINGS_PATH)


def load_settings() -> dict[str, str | float | int | bool]:
    options = list_model_options()
    settings = DEFAULT_SETTINGS.copy()

    with SETTINGS_LOCK:
        if SETTINGS_PATH.exists():
            try:
                stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    settings.update(
                        {
                            key: value
                            for key, value in stored.items()
                            if isinstance(value, (str, int, float))
                        }
                    )
            except json.JSONDecodeError as exc:
                print(f"Failed to parse settings.json: {exc}")

    if settings["yolo_model"] not in options["yolo"] and options["yolo"]:
        settings["yolo_model"] = options["yolo"][0]
    if settings["cnn_model"] not in options["cnn"] and options["cnn"]:
        settings["cnn_model"] = options["cnn"][0]

    if not SETTINGS_PATH.exists():
        save_settings(settings)

    return settings


def ensure_valid_settings(selected: dict[str, str | bool]) -> tuple[dict[str, str | float | int | bool] | None, str | None]:
    options = list_model_options()

    yolo_model = selected.get("yolo_model", "").strip()
    cnn_model = selected.get("cnn_model", "").strip()
    yolo_conf_raw = str(selected.get("yolo_conf", "")).strip()
    cnn_conf_raw = str(selected.get("cnn_conf", "")).strip()
    rembg_model = str(selected.get("rembg_model", "")).strip()
    rembg_alpha_matting = bool(selected.get("rembg_alpha_matting", False))
    rembg_fg_raw = str(selected.get("rembg_alpha_matting_foreground_threshold", "")).strip()
    rembg_bg_raw = str(selected.get("rembg_alpha_matting_background_threshold", "")).strip()
    rembg_erode_raw = str(selected.get("rembg_alpha_matting_erode_size", "")).strip()

    if yolo_model not in options["yolo"]:
        return None, "頭部検出モデルの選択が不正です。"
    if cnn_model not in options["cnn"]:
        return None, "画像分類モデルの選択が不正です。"
    if rembg_model not in REMBG_MODEL_OPTIONS:
        return None, "rembg モデルの選択が不正です。"
    try:
        yolo_conf = float(yolo_conf_raw)
    except ValueError:
        return None, "YOLO のしきい値は数値で指定してください。"
    try:
        cnn_conf = float(cnn_conf_raw)
    except ValueError:
        return None, "CNN のしきい値は数値で指定してください。"
    try:
        rembg_fg = int(rembg_fg_raw)
    except ValueError:
        return None, "rembg 前景しきい値は整数で指定してください。"
    try:
        rembg_bg = int(rembg_bg_raw)
    except ValueError:
        return None, "rembg 背景しきい値は整数で指定してください。"
    try:
        rembg_erode = int(rembg_erode_raw)
    except ValueError:
        return None, "rembg erode size は整数で指定してください。"
    if not 0 <= yolo_conf <= 1:
        return None, "YOLO のしきい値は 0.0 から 1.0 の範囲で指定してください。"
    if not 0 <= cnn_conf <= 1:
        return None, "CNN のしきい値は 0.0 から 1.0 の範囲で指定してください。"
    if not 0 <= rembg_fg <= 255:
        return None, "rembg 前景しきい値は 0 から 255 の範囲で指定してください。"
    if not 0 <= rembg_bg <= 255:
        return None, "rembg 背景しきい値は 0 から 255 の範囲で指定してください。"
    if rembg_erode < 0:
        return None, "rembg erode size は 0 以上で指定してください。"

    return {
        "yolo_model": yolo_model,
        "cnn_model": cnn_model,
        "yolo_conf": yolo_conf,
        "cnn_conf": cnn_conf,
        "rembg_model": rembg_model,
        "rembg_alpha_matting": rembg_alpha_matting,
        "rembg_alpha_matting_foreground_threshold": rembg_fg,
        "rembg_alpha_matting_background_threshold": rembg_bg,
        "rembg_alpha_matting_erode_size": rembg_erode,
    }, None


def safe_remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        pass


def link_path(source: Path, destination: Path, is_dir: bool = False) -> None:
    safe_remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=is_dir)


def replace_text_in_file(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def replace_regex_in_file(path: Path, pattern: str, repl: str, flags: int = re.MULTILINE) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected pattern not found in {path}: {pattern}")
    path.write_text(updated, encoding="utf-8")


def configure_workspace_thresholds(workspace: Path, settings: dict[str, str | float]) -> None:
    yolo_conf = float(settings["yolo_conf"])

    replace_text_in_file(
        workspace / "muscut_tools" / "muscut_default.py",
        "            yolo_conf = 0.5\n",
        f"            yolo_conf = {yolo_conf}\n",
    )
    replace_text_in_file(
        workspace / "muscut_tools" / "muscut_extract_ok_frames.py",
        "            yolo_conf = 0.5\n",
        f"            yolo_conf = {yolo_conf}\n",
    )
    replace_text_in_file(
        workspace / "focus_threshold_checker.py",
        "        conf=0.5,\n",
        f"        conf={yolo_conf},\n",
    )


def configure_workspace_rembg_settings(workspace: Path, settings: dict[str, str | float | bool]) -> None:
    rembg_model = str(settings["rembg_model"])
    alpha_matting = "True" if bool(settings["rembg_alpha_matting"]) else "False"
    foreground_threshold = int(settings["rembg_alpha_matting_foreground_threshold"])
    background_threshold = int(settings["rembg_alpha_matting_background_threshold"])
    erode_size = int(settings["rembg_alpha_matting_erode_size"])

    for relative_path in (
        ("muscut_tools", "muscut_rembg_multi_process.py"),
        ("muscut_tools", "muscut_rembg.py"),
    ):
        replace_regex_in_file(
            workspace / relative_path[0] / relative_path[1],
            r'unet_model_name = ".*?"',
            f'unet_model_name = "{rembg_model}"',
        )

    replace_regex_in_file(
        workspace / "muscut_functions" / "rembg_functions.py",
        r"def remove_bg\(image, session, file_name\):\n(?:    .*\n)+?    return rembg_img, file_name",
        (
            "def remove_bg(image, session, file_name):\n"
            "    rembg_img = remove(\n"
            "        image,\n"
            f"        alpha_matting={alpha_matting},\n"
            f"        alpha_matting_foreground_threshold={foreground_threshold},\n"
            f"        alpha_matting_background_threshold={background_threshold},\n"
            f"        alpha_matting_erode_size={erode_size},\n"
            "        session=session,\n"
            "    )\n"
            "    return rembg_img, file_name"
        ),
    )

    replace_regex_in_file(
        workspace / "muscut_tools" / "muscut_rembg.py",
        r"output = remove\([\s\S]*?session=session,\n\s*\)",
        (
            "output = remove(\n"
            "            input,\n"
            f"            alpha_matting={alpha_matting},\n"
            f"            alpha_matting_foreground_threshold={foreground_threshold},\n"
            f"            alpha_matting_background_threshold={background_threshold},\n"
            f"            alpha_matting_erode_size={erode_size},\n"
            "            session=session,\n"
            "        )"
        ),
        flags=re.MULTILINE | re.DOTALL,
    )


def configure_workspace_rembg_runtime(workspace: Path, settings: dict[str, str | float | bool]) -> None:
    rembg_model = str(settings["rembg_model"])

    replace_text_in_file(
        workspace / "muscut_tools" / "muscut_rembg_multi_process.py",
        'os_name = platform.system()\n\n\ndef main(input_path):\n',
        (
            'os_name = platform.system()\n\n\n'
            'def is_onnx_cuda_oom(exc: Exception) -> bool:\n'
            '    message = str(exc).lower()\n'
            '    return "onnxruntimeerror" in message and "cuda failure 2" in message and "out of memory" in message\n\n\n'
            'def build_rembg_session(unet_model_name: str, prefer_gpu: bool = True):\n'
            '    session_options = ort.SessionOptions()\n\n'
            '    if os_name == "Darwin":\n'
            '        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]\n'
            '    elif prefer_gpu:\n'
            '        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]\n'
            '    else:\n'
            '        providers = ["CPUExecutionProvider"]\n\n'
            '    session = new_session(\n'
            '        unet_model_name,\n'
            '        sess_options=session_options,\n'
            '        providers=providers,\n'
            '    )\n\n'
            '    if "CUDAExecutionProvider" in session.providers:\n'
            '        print("\\033[32mONNX Runtime is using GPU.\\033[0m")\n'
            '    elif "CoreMLExecutionProvider" in session.providers:\n'
            '        print("\\033[32mONNX Runtime is using CoreML.\\033[0m")\n'
            '    else:\n'
            '        print("\\033[33mONNX Runtime is using CPU.\\033[0m")\n\n'
            '    return session\n\n\n'
            'def main(input_path):\n'
        ),
    )

    replace_regex_in_file(
        workspace / "muscut_tools" / "muscut_rembg_multi_process.py",
        r'    print\("\\033\[32m背景除去開始\\033\[0m"\)\n[\s\S]*?    rembg_images, file_names = rembg_functions\.process_rembg\(\n        images, imgnames, session\n    \)\n',
        (
            '    print("\\033[32m背景除去開始\\033[0m")\n'
            f'    unet_model_name = "{rembg_model}"\n'
            '    print(f"\\033[32m start rembg model:{unet_model_name}\\033[0m")\n\n'
            '    session = build_rembg_session(unet_model_name, prefer_gpu=(os_name != "Darwin"))\n\n'
            '    try:\n'
            '        rembg_images, file_names = rembg_functions.process_rembg(images, imgnames, session)\n'
            '    except Exception as exc:\n'
            '        if not is_onnx_cuda_oom(exc) or os_name == "Darwin":\n'
            '            raise\n\n'
            '        print("\\033[33mGPU メモリ不足のため rembg を CPU にフォールバックします。\\033[0m")\n'
            '        del session\n'
            '        gc.collect()\n'
            '        rembg_images, file_names = rembg_functions.process_rembg(\n'
            '            images,\n'
            '            imgnames,\n'
            '            build_rembg_session(unet_model_name, prefer_gpu=False),\n'
            '        )\n'
        ),
        flags=re.MULTILINE | re.DOTALL,
    )


def create_model_workspace(job_token: str, settings: dict[str, str | float | int | bool]) -> Path:
    workspace = WORKSPACE_DIR / job_token
    safe_rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    shutil.copy2(REPO_ROOT / "muscut.py", workspace / "muscut.py")
    shutil.copy2(REPO_ROOT / "muscut_with_rembg.py", workspace / "muscut_with_rembg.py")
    shutil.copy2(REPO_ROOT / "focus_threshold_checker.py", workspace / "focus_threshold_checker.py")
    shutil.copytree(REPO_ROOT / "muscut_tools", workspace / "muscut_tools", dirs_exist_ok=True)
    shutil.copytree(REPO_ROOT / "muscut_functions", workspace / "muscut_functions", dirs_exist_ok=True)

    models_dir = workspace / "muscut_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_yolo = REPO_ROOT / settings["yolo_model"]
    selected_cnn = REPO_ROOT / settings["cnn_model"]

    if SYSTEM_NAME == "Darwin":
        link_path(selected_yolo, models_dir / "yolo.mlmodel")
        link_path(REPO_ROOT / settings["cnn_model"], models_dir / "ct_cnn.mlmodel")
    else:
        link_path(selected_yolo, models_dir / "yolo.pt")
        cnn_parent = models_dir / "cnn"
        cnn_parent.mkdir(parents=True, exist_ok=True)
        link_path(selected_cnn, cnn_parent / "savedmodel", is_dir=True)

    configure_workspace_thresholds(workspace, settings)
    configure_workspace_rembg_settings(workspace, settings)
    configure_workspace_rembg_runtime(workspace, settings)
    return workspace


def open_directory_in_file_manager(path: Path) -> tuple[bool, str]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return False, f"フォルダが見つかりません: {path}"

    try:
        if SYSTEM_NAME == "Darwin":
            subprocess.Popen(["open", str(resolved)])
        elif SYSTEM_NAME == "Windows":
            subprocess.Popen(["explorer", str(resolved)])
        else:
            subprocess.Popen(["xdg-open", str(resolved)])
    except OSError as exc:
        return False, f"フォルダを開けませんでした: {exc}"

    return True, f"フォルダを開きました: {to_repo_relative(resolved)}"


def build_command(
    script_path: Path,
    upload_path: Path,
    image_format: str,
    extract_count: int | None,
    cnn_conf: float | None,
    pint_threshold: int | None,
    all_extract: bool,
    tool: str,
    without_cnn: bool,
    dev_flag: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        "-f",
        str(upload_path),
        "-i",
        image_format,
    ]
    if extract_count is not None:
        command.extend(["-n", str(extract_count)])
    if cnn_conf is not None:
        command.extend(["-c", str(cnn_conf)])
    if pint_threshold is not None:
        command.extend(["-p", str(pint_threshold)])
    if all_extract:
        command.append("-a")
    command.extend(["-t", tool])
    if without_cnn:
        command.append("-wc")
    if dev_flag:
        command.append("-dev")
    return command


def zip_output_dir(source_dir: Path, destination_zip: Path) -> Path:
    archive_base = destination_zip.with_suffix("")
    created = shutil.make_archive(str(archive_base), "zip", str(source_dir))
    return Path(created)


def count_output_files(output_dir: Path) -> int:
    return sum(1 for path in output_dir.rglob("*") if path.is_file())


def build_focus_preview_command(workspace: Path, movie_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(APP_DIR / "focus_preview.py"),
        "--movie",
        str(movie_path),
        "--output",
        str(output_path),
        "--num-images",
        "15",
        "--batch-size",
        "8",
    ]


def build_model_check_command(
    workspace: Path,
    movie_path: Path,
    output_path: Path,
    yolo_conf: float,
    cnn_conf: float,
    pint_threshold: int,
) -> list[str]:
    return [
        sys.executable,
        str(APP_DIR / "model_check_preview.py"),
        "--movie",
        str(movie_path),
        "--output",
        str(output_path),
        "--yolo-conf",
        str(yolo_conf),
        "--cnn-conf",
        str(cnn_conf),
        "--pint-threshold",
        str(pint_threshold),
    ]


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def safe_rmtree(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        pass


def mark_download_consumed(filename: str) -> None:
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("download_name") == filename:
                job["download_name"] = None
                if job["status"] == "completed":
                    job["phase"] = "ダウンロード済みです。"


def mark_multiple_downloads_consumed(filenames: list[str]) -> None:
    target = set(filenames)
    if not target:
        return
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("download_name") in target:
                job["download_name"] = None
                if job["status"] == "completed":
                    job["phase"] = "一括ダウンロード済みです。"


def collect_bulk_download_items_locked() -> list[tuple[str, str, Path, str]]:
    completed_items: list[tuple[str, str, Path, str]] = []
    for job_id, job in JOBS.items():
        if job["status"] != "completed":
            continue
        download_name = job.get("download_name")
        if not download_name:
            continue
        zip_path = DOWNLOAD_DIR / download_name
        if zip_path.exists():
            completed_items.append((job_id, job["file_name"], zip_path, download_name))
    return completed_items


def build_unique_bulk_archive_name(file_name: str, job_id: str, used_names: set[str]) -> str:
    stem = sanitize_filename(Path(file_name).stem).strip(".") or "result"
    archive_name = f"{stem}.zip"
    if archive_name in used_names:
        archive_name = f"{stem}-{job_id[-8:]}.zip"
    suffix = 2
    while archive_name in used_names:
        archive_name = f"{stem}-{job_id[-8:]}-{suffix}.zip"
        suffix += 1
    used_names.add(archive_name)
    return archive_name


def build_bulk_download_zip() -> tuple[Path | None, int, list[Path], list[str]]:
    with JOBS_LOCK:
        completed_items = collect_bulk_download_items_locked()

    if not completed_items:
        return None, 0, [], []

    bundle_name = f"bulk-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.zip"
    bundle_path = TMP_DIR / bundle_name
    safe_unlink(bundle_path)

    zip_paths: list[Path] = []
    filenames: list[str] = []
    used_archive_names: set[str] = set()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job_id, file_name, item_path, download_name in completed_items:
            if not item_path.exists():
                continue
            archive.write(item_path, arcname=build_unique_bulk_archive_name(file_name, job_id, used_archive_names))
            zip_paths.append(item_path)
            filenames.append(download_name)

    if not zip_paths:
        safe_unlink(bundle_path)
        return None, 0, [], []

    return bundle_path, len(zip_paths), zip_paths, filenames


def append_log(job: dict, line: str) -> None:
    stripped = line.rstrip()
    if stripped:
        job["logs"].append(stripped)


def update_progress_from_line(job: dict, line: str) -> None:
    if "顔検出完了" in line:
        job["phase"] = "顔検出が終わりました。画像を整理しています。"
        job["progress_percent"] = max(job["progress_percent"], 95)
        return
    if "一時作業フォルダーを作成しました" in line:
        job["phase"] = "クラスタリングを開始しています。"
        job["progress_percent"] = max(job["progress_percent"], 96)
        return
    if "背景除去開始" in line:
        job["phase"] = "背景除去を行っています。"
        job["progress_percent"] = max(job["progress_percent"], 97)
        return
    if "切り取り中" in line:
        job["phase"] = "背景除去後の切り取りを行っています。"
        job["progress_percent"] = max(job["progress_percent"], 98)
        return
    if "All Done!" in line or "処理が完了しました" in line or ": Done!" in line:
        job["phase"] = "ダウンロードの準備をしています。"
        job["progress_percent"] = max(job["progress_percent"], 99)
        return

    percent_match = PROGRESS_PERCENT_RE.search(line)
    count_match = PROGRESS_COUNT_RE.search(line)
    if percent_match:
        percent = max(0, min(100, int(percent_match.group(1))))
        job["phase"] = "フレームを解析しています。"
        job["progress_percent"] = percent

    if count_match:
        current = count_match.group(1)
        total = count_match.group(2)
        if job["phase"] in {"待機中です。", "順番待ちです。"}:
            job["phase"] = "フレームを解析しています。"
        job["progress_text"] = f"{current} / {total}"


def build_preview_payload_locked(job: dict) -> dict:
    return {
        "preview_id": job["preview_id"],
        "file_name": job["file_name"],
        "status": job["status"],
        "phase": job["phase"],
        "progress_percent": job["progress_percent"],
        "progress_text": job["progress_text"],
        "logs": list(job["logs"]),
        "error_message": job["error_message"],
        "image_url": (
            url_for("focus_preview_image", preview_id=job["preview_id"])
            if job["status"] == "completed" and job["preview_path"].exists()
            else None
        ),
    }


def update_preview_progress_from_line(job: dict, line: str) -> None:
    if "Starting Head Detection" in line:
        job["phase"] = "頭部を検出しています。"
        job["progress_percent"] = max(job["progress_percent"], 5)
        return
    if "Head detection completed" in line:
        job["phase"] = "頭部検出が終わりました。候補画像を整理しています。"
        job["progress_percent"] = max(job["progress_percent"], 40)
        return
    if "Number of head detections" in line:
        job["phase"] = "候補画像を整理しています。"
        job["progress_percent"] = max(job["progress_percent"], 45)
        return
    if "Starting Side-Profile Classification" in line:
        job["phase"] = "向きを判定しています。"
        job["progress_percent"] = max(job["progress_percent"], 75)
        return
    if "Number of side-profile classifications" in line:
        job["phase"] = "参照画像をまとめています。"
        job["progress_percent"] = max(job["progress_percent"], 90)
        return

    count_match = PROGRESS_COUNT_RE.search(line)
    if count_match:
        current_raw = float(count_match.group(1))
        total_raw = float(count_match.group(2))
        current = int(current_raw) if current_raw.is_integer() else current_raw
        total = int(total_raw) if total_raw.is_integer() else total_raw
        job["progress_text"] = f"{current} / {total}"

        if total_raw > 0:
            ratio = max(0.0, min(1.0, current_raw / total_raw))
            if "Processing Crps" in line:
                job["phase"] = "候補画像を整理しています。"
                weighted_percent = 45 + (ratio * 30)
                job["progress_percent"] = max(job["progress_percent"], int(weighted_percent))
            elif "predict" in line or "ms/step" in line:
                job["phase"] = "向きを判定しています。"
                weighted_percent = 75 + (ratio * 15)
                job["progress_percent"] = max(job["progress_percent"], int(weighted_percent))


def build_model_check_payload_locked(job: dict) -> dict:
    return {
        "check_id": job["check_id"],
        "file_name": job["file_name"],
        "status": job["status"],
        "phase": job["phase"],
        "progress_percent": job["progress_percent"],
        "progress_text": job["progress_text"],
        "logs": list(job["logs"]),
        "error_message": job["error_message"],
        "video_url": (
            url_for("model_check_video", check_id=job["check_id"])
            if job["status"] == "completed" and job["video_path"].exists()
            else None
        ),
        "download_url": (
            url_for("model_check_download", check_id=job["check_id"])
            if job["status"] == "completed" and job["video_path"].exists()
            else None
        ),
        "selected_yolo_model_label": job.get("selected_yolo_model_label"),
        "selected_cnn_model_label": job.get("selected_cnn_model_label"),
        "yolo_conf": job.get("yolo_conf"),
        "cnn_conf": job.get("cnn_conf"),
        "pint_threshold": job.get("pint_threshold"),
    }


def cleanup_model_check_job(check_id: str, terminate_running: bool = True) -> bool:
    with MODEL_CHECK_JOBS_LOCK:
        job = MODEL_CHECK_JOBS.get(check_id)
        if job is None:
            return False
        pid = job.get("pid")
        upload_path = job["upload_path"]
        video_path = job["video_path"]
        workspace_dir = job["workspace_dir"]
        status = job["status"]
        if status in {"running", "queued"} and not terminate_running:
            return False
        MODEL_CHECK_JOBS.pop(check_id, None)

    if terminate_running and pid:
        try:
            os.kill(pid, 15)
        except OSError:
            pass

    safe_unlink(upload_path)
    safe_unlink(video_path)
    safe_rmtree(workspace_dir)
    return True


def update_model_check_progress_from_line(job: dict, line: str) -> None:
    if line.startswith("MODEL_CHECK_PHASE "):
        job["phase"] = line.removeprefix("MODEL_CHECK_PHASE ").strip()
        return

    if line.startswith("MODEL_CHECK_PROGRESS "):
        progress_body = line.removeprefix("MODEL_CHECK_PROGRESS ").strip()
        parts = progress_body.split()
        if parts:
            counts = parts[0]
            if "/" in counts:
                current_text, total_text = counts.split("/", 1)
                try:
                    current = float(current_text)
                    total = float(total_text)
                except ValueError:
                    return
                if total > 0:
                    ratio = max(0.0, min(1.0, current / total))
                    job["progress_percent"] = max(job["progress_percent"], int(ratio * 100))
                    current_value = int(current) if current.is_integer() else current
                    total_value = int(total) if total.is_integer() else total
                    job["progress_text"] = f"{current_value} / {total_value}"
        return

    if line.startswith("MODEL_CHECK_FPS "):
        fps_value = line.removeprefix("MODEL_CHECK_FPS ").strip()
        if fps_value:
            existing = job["progress_text"].strip()
            job["progress_text"] = f"{existing} / {fps_value} fps" if existing else f"{fps_value} fps"
        return


def execute_model_check_job(check_id: str) -> None:
    with MODEL_CHECK_JOBS_LOCK:
        job = MODEL_CHECK_JOBS.get(check_id)
        if job is None:
            return
        upload_path = job["upload_path"]
        video_path = job["video_path"]
        workspace_dir = job["workspace_dir"]

    try:
        process = subprocess.Popen(
            build_model_check_command(
                workspace_dir,
                upload_path,
                video_path,
                float(job["yolo_conf"]),
                float(job["cnn_conf"]),
                int(job["pint_threshold"]),
            ),
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        with MODEL_CHECK_JOBS_LOCK:
            job = MODEL_CHECK_JOBS.get(check_id)
            if job is None:
                return
            job["pid"] = process.pid
            job["status"] = "running"
            job["phase"] = "確認動画の生成を開始しました。"

        assert process.stdout is not None
        for raw_line in process.stdout:
            with MODEL_CHECK_JOBS_LOCK:
                job = MODEL_CHECK_JOBS.get(check_id)
                if job is None:
                    continue
                append_log(job, raw_line)
                update_model_check_progress_from_line(job, raw_line.rstrip())

        return_code = process.wait()
        with MODEL_CHECK_JOBS_LOCK:
            job = MODEL_CHECK_JOBS.get(check_id)
            if job is None:
                return
            if return_code != 0:
                job["status"] = "error"
                job["phase"] = "確認動画の生成に失敗しました。"
                job["error_message"] = "\n".join(job["logs"]) or "詳細不明のエラーです。"
                safe_unlink(video_path)
            elif not video_path.exists():
                job["status"] = "error"
                job["phase"] = "確認動画が見つかりません。"
                job["error_message"] = "処理は終了しましたが、確認動画が生成されませんでした。"
            else:
                job["status"] = "completed"
                job["phase"] = "確認動画を再生できます。"
                job["progress_percent"] = 100
    except Exception as exc:
        with MODEL_CHECK_JOBS_LOCK:
            job = MODEL_CHECK_JOBS.get(check_id)
            if job is not None:
                job["status"] = "error"
                job["phase"] = "内部エラーで停止しました。"
                job["error_message"] = str(exc)
                safe_unlink(job["video_path"])
    finally:
        safe_unlink(upload_path)
        safe_rmtree(workspace_dir)


def execute_preview_job(preview_id: str) -> None:
    with PREVIEW_JOBS_LOCK:
        job = PREVIEW_JOBS.get(preview_id)
        if job is None:
            return
        upload_path = job["upload_path"]
        preview_path = job["preview_path"]
        workspace_dir = job["workspace_dir"]

    try:
        process = subprocess.Popen(
            build_focus_preview_command(workspace_dir, upload_path, preview_path),
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        with PREVIEW_JOBS_LOCK:
            job = PREVIEW_JOBS.get(preview_id)
            if job is None:
                return
            job["pid"] = process.pid
            job["status"] = "running"
            job["phase"] = "参照画像の生成を開始しました。"

        assert process.stdout is not None
        for raw_line in process.stdout:
            with PREVIEW_JOBS_LOCK:
                job = PREVIEW_JOBS.get(preview_id)
                if job is None:
                    continue
                append_log(job, raw_line)
                update_preview_progress_from_line(job, raw_line)

        return_code = process.wait()
        with PREVIEW_JOBS_LOCK:
            job = PREVIEW_JOBS.get(preview_id)
            if job is None:
                return
            if return_code != 0:
                job["status"] = "error"
                job["phase"] = "参照画像の生成に失敗しました。"
                job["error_message"] = "\n".join(job["logs"]) or "詳細不明のエラーです。"
                safe_unlink(preview_path)
            elif not preview_path.exists():
                job["status"] = "error"
                job["phase"] = "参照画像が見つかりません。"
                job["error_message"] = "処理は終了しましたが、参照画像ファイルが生成されませんでした。"
            else:
                job["status"] = "completed"
                job["phase"] = "参照画像を表示できます。"
                job["progress_percent"] = 100
    except Exception as exc:
        with PREVIEW_JOBS_LOCK:
            job = PREVIEW_JOBS.get(preview_id)
            if job is not None:
                job["status"] = "error"
                job["phase"] = "内部エラーで停止しました。"
                job["error_message"] = str(exc)
                safe_unlink(job["preview_path"])
    finally:
        safe_unlink(upload_path)
        safe_rmtree(workspace_dir)


def refresh_queue_positions_locked() -> None:
    for index, queued_job_id in enumerate(JOB_QUEUE, start=1):
        job = JOBS.get(queued_job_id)
        if job is None:
            continue
        job["queue_position"] = index
        job["phase"] = f"順番待ちです。前に {index - 1} 件あります。"


def collect_system_counts_locked() -> tuple[int, int]:
    return len(ACTIVE_JOB_IDS), len(JOB_QUEUE)


def count_active_jobs_by_mode_locked() -> dict[str, int]:
    counts = {mode: 0 for mode in MODE_CONCURRENCY_LIMITS}
    for job_id in ACTIVE_JOB_IDS:
        job = JOBS.get(job_id)
        if job is None:
            continue
        counts[job["processing_mode"]] = counts.get(job["processing_mode"], 0) + 1
    return counts


def can_start_job_locked(job: dict) -> bool:
    active_total = len(ACTIVE_JOB_IDS)
    if active_total >= MAX_TOTAL_CONCURRENT_JOBS:
        return False
    active_by_mode = count_active_jobs_by_mode_locked()
    mode = job["processing_mode"]
    return active_by_mode.get(mode, 0) < MODE_CONCURRENCY_LIMITS[mode]


def build_job_payload_locked(job: dict) -> dict:
    active_count, queued_count = collect_system_counts_locked()
    active_by_mode = count_active_jobs_by_mode_locked()
    return {
        "job_id": job["job_id"],
        "file_name": job["file_name"],
        "status": job["status"],
        "phase": job["phase"],
        "progress_percent": job["progress_percent"],
        "progress_text": job["progress_text"],
        "image_count": job["image_count"],
        "processing_mode": job["processing_mode"],
        "selected_yolo_model": job.get("selected_yolo_model"),
        "selected_cnn_model": job.get("selected_cnn_model"),
        "selected_yolo_model_label": job.get("selected_yolo_model_label"),
        "selected_cnn_model_label": job.get("selected_cnn_model_label"),
        "yolo_conf": job.get("yolo_conf"),
        "cnn_conf": job.get("cnn_conf"),
        "error_message": job["error_message"],
        "logs": list(job["logs"]),
        "queue_position": job["queue_position"],
        "active_count": active_count,
        "queued_count": queued_count,
        "active_standard_count": active_by_mode["standard"],
        "active_rembg_count": active_by_mode["rembg"],
        "max_total_concurrent": MAX_TOTAL_CONCURRENT_JOBS,
        "max_standard_concurrent": MODE_CONCURRENCY_LIMITS["standard"],
        "max_rembg_concurrent": MODE_CONCURRENCY_LIMITS["rembg"],
        "download_url": (
            url_for("download_result", filename=job["download_name"])
            if job["download_name"]
            else None
        ),
    }


def schedule_jobs() -> None:
    to_start: list[str] = []
    with JOBS_LOCK:
        if JOB_QUEUE:
            remaining_queue = deque()
            while JOB_QUEUE and len(ACTIVE_JOB_IDS) < MAX_TOTAL_CONCURRENT_JOBS:
                job_id = JOB_QUEUE.popleft()
                job = JOBS.get(job_id)
                if job is None or job["status"] != "queued":
                    continue
                if not can_start_job_locked(job):
                    remaining_queue.append(job_id)
                    continue
                ACTIVE_JOB_IDS.add(job_id)
                job["status"] = "running"
                job["queue_position"] = 0
                job["phase"] = "処理を開始しました。"
                to_start.append(job_id)
            while JOB_QUEUE:
                remaining_queue.append(JOB_QUEUE.popleft())
            JOB_QUEUE.extend(remaining_queue)
        refresh_queue_positions_locked()

    for job_id in to_start:
        worker = threading.Thread(target=execute_job, args=(job_id,), daemon=True)
        worker.start()


def finalize_job_slot(job_id: str) -> None:
    with JOBS_LOCK:
        ACTIVE_JOB_IDS.discard(job_id)
    schedule_jobs()


def execute_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        command = job["command_list"]
        workspace_dir = job["workspace_dir"]
        output_dir = job["output_dir"]
        output_root = job["output_root"]
        zip_path = job["zip_path"]
        upload_path = job["upload_path"]

    try:
        process = subprocess.Popen(
            command,
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        with JOBS_LOCK:
            job = JOBS[job_id]
            job["pid"] = process.pid

        assert process.stdout is not None
        for raw_line in process.stdout:
            with JOBS_LOCK:
                job = JOBS[job_id]
                append_log(job, raw_line)
                update_progress_from_line(job, raw_line)

        return_code = process.wait()

        if return_code != 0:
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "error"
                job["phase"] = "エラーで停止しました。"
                job["error_message"] = "\n".join(job["logs"]) or "詳細不明のエラーです。"
            return

        if not output_dir.exists():
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "error"
                job["phase"] = "出力画像が見つかりません。"
                job["error_message"] = "処理は終了しましたが、期待した出力フォルダが作成されませんでした。"
            return

        created_zip = zip_output_dir(output_dir, zip_path)
        image_count = count_output_files(output_dir)
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "completed"
            job["phase"] = "完了しました。"
            job["progress_percent"] = 100
            job["download_name"] = created_zip.name
            job["image_count"] = image_count
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["phase"] = "内部エラーで停止しました。"
                job["error_message"] = str(exc)
    finally:
        safe_unlink(upload_path)
        safe_rmtree(output_root)
        safe_rmtree(workspace_dir)
        finalize_job_slot(job_id)


def validate_request_form():
    upload = request.files.get("movie_file")
    if upload is None or upload.filename == "":
        return None, ("動画ファイルを選択してください。", 400)

    image_format = request.form.get("image_format", "png").lower()
    if image_format not in {"png", "jpg"}:
        return None, ("画像形式は png または jpg を指定してください。", 400)

    processing_mode = request.form.get("processing_mode", "standard").strip() or "standard"
    if processing_mode not in {"standard", "rembg"}:
        return None, ("処理モードが不正です。", 400)

    tool = request.form.get("tool", "default").strip() or "default"
    if processing_mode == "rembg":
        tool = "extract_ok_frames"
    elif tool not in {"default", "kmeans_image_extractor"}:
        return None, ("ツール指定が不正です。", 400)

    all_extract = request.form.get("all_extract") == "on"
    without_cnn = request.form.get("without_cnn") == "on"
    dev_flag = request.form.get("dev_flag") == "on"

    extract_count = None
    if not all_extract:
        try:
            extract_count = int(request.form.get("extract_count", "").strip())
        except ValueError:
            return None, ("抽出枚数には整数を指定してください。", 400)
        if extract_count < 1:
            return None, ("抽出枚数は 1 以上を指定してください。", 400)

    pint_threshold = None
    pint_raw = request.form.get("pint_threshold", "").strip()
    if pint_raw:
        try:
            pint_threshold = int(pint_raw)
        except ValueError:
            return None, ("ピント閾値には整数を指定してください。", 400)
        if pint_threshold < 1:
            return None, ("ピント閾値は 1 以上を指定してください。", 400)

    return {
        "upload": upload,
        "processing_mode": processing_mode,
        "image_format": image_format,
        "tool": tool,
        "all_extract": all_extract,
        "without_cnn": without_cnn,
        "dev_flag": dev_flag,
        "extract_count": extract_count,
        "pint_threshold": pint_threshold,
    }, None


@app.get("/")
def index():
    settings = load_settings()
    model_choices = build_model_choices()
    yolo_label_map = {item["value"]: item["label"] for item in model_choices["yolo"]}
    cnn_label_map = {item["value"]: item["label"] for item in model_choices["cnn"]}
    return render_template(
        "index.html",
        max_total_concurrent=MAX_TOTAL_CONCURRENT_JOBS,
        max_standard_concurrent=MODE_CONCURRENCY_LIMITS["standard"],
        max_rembg_concurrent=MODE_CONCURRENCY_LIMITS["rembg"],
        selected_yolo_model=settings["yolo_model"],
        selected_cnn_model=settings["cnn_model"],
        selected_yolo_model_label=yolo_label_map.get(settings["yolo_model"], settings["yolo_model"]),
        selected_cnn_model_label=cnn_label_map.get(settings["cnn_model"], settings["cnn_model"]),
        yolo_conf=settings["yolo_conf"],
        cnn_conf=settings["cnn_conf"],
    )


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    message = request.args.get("message", "")
    error_message = request.args.get("error", "")

    if request.method == "GET" and request.args.get("reloaded") == "1":
        message = "モデル候補を再読込しました。新しく配置したカスタムモデルがあれば一覧に反映されます。"

    if request.method == "POST":
        validated, error_message = ensure_valid_settings(
            {
                "yolo_model": request.form.get("yolo_model", ""),
                "cnn_model": request.form.get("cnn_model", ""),
                "yolo_conf": request.form.get("yolo_conf", ""),
                "cnn_conf": request.form.get("cnn_conf", ""),
                "rembg_model": request.form.get("rembg_model", ""),
                "rembg_alpha_matting": request.form.get("rembg_alpha_matting") == "on",
                "rembg_alpha_matting_foreground_threshold": request.form.get("rembg_alpha_matting_foreground_threshold", ""),
                "rembg_alpha_matting_background_threshold": request.form.get("rembg_alpha_matting_background_threshold", ""),
                "rembg_alpha_matting_erode_size": request.form.get("rembg_alpha_matting_erode_size", ""),
            }
        )
        if validated is not None:
            save_settings(validated)
            message = "設定を保存しました。新しく開始する処理から反映されます。"

    current_settings = load_settings()
    choices = build_model_choices()
    return render_template(
        "settings.html",
        yolo_choices=choices["yolo"],
        cnn_choices=choices["cnn"],
        selected_yolo_model=current_settings["yolo_model"],
        selected_cnn_model=current_settings["cnn_model"],
        yolo_conf=current_settings["yolo_conf"],
        cnn_conf=current_settings["cnn_conf"],
        rembg_model=current_settings["rembg_model"],
        rembg_alpha_matting=current_settings["rembg_alpha_matting"],
        rembg_alpha_matting_foreground_threshold=current_settings["rembg_alpha_matting_foreground_threshold"],
        rembg_alpha_matting_background_threshold=current_settings["rembg_alpha_matting_background_threshold"],
        rembg_alpha_matting_erode_size=current_settings["rembg_alpha_matting_erode_size"],
        rembg_model_choices=REMBG_MODEL_OPTIONS,
        default_yolo_model=DEFAULT_SETTINGS["yolo_model"],
        default_cnn_model=DEFAULT_SETTINGS["cnn_model"],
        default_yolo_conf=DEFAULT_SETTINGS["yolo_conf"],
        default_cnn_conf=DEFAULT_SETTINGS["cnn_conf"],
        default_rembg_model=DEFAULT_SETTINGS["rembg_model"],
        default_rembg_alpha_matting=DEFAULT_SETTINGS["rembg_alpha_matting"],
        default_rembg_alpha_matting_foreground_threshold=DEFAULT_SETTINGS["rembg_alpha_matting_foreground_threshold"],
        default_rembg_alpha_matting_background_threshold=DEFAULT_SETTINGS["rembg_alpha_matting_background_threshold"],
        default_rembg_alpha_matting_erode_size=DEFAULT_SETTINGS["rembg_alpha_matting_erode_size"],
        custom_yolo_dir=to_repo_relative(CUSTOM_YOLO_DIR),
        custom_cnn_dir=to_repo_relative(CUSTOM_CNN_DIR),
        message=message,
        error_message=error_message,
    )


@app.post("/settings/open-model-root/<model_kind>")
def open_model_root(model_kind: str):
    if model_kind == "yolo":
        target_dir = CUSTOM_YOLO_DIR
    elif model_kind == "cnn":
        target_dir = CUSTOM_CNN_DIR
    else:
        abort(404)

    opened, feedback = open_directory_in_file_manager(target_dir)
    if opened:
        return redirect(url_for("settings_page", message=feedback))
    return redirect(url_for("settings_page", error=feedback))


@app.get("/model-check")
def model_check_page():
    settings = load_settings()
    model_choices = build_model_choices()
    yolo_label_map = {item["value"]: item["label"] for item in model_choices["yolo"]}
    cnn_label_map = {item["value"]: item["label"] for item in model_choices["cnn"]}
    return render_template(
        "model_check.html",
        selected_yolo_model_label=yolo_label_map.get(settings["yolo_model"], settings["yolo_model"]),
        selected_cnn_model_label=cnn_label_map.get(settings["cnn_model"], settings["cnn_model"]),
        yolo_conf=settings["yolo_conf"],
        cnn_conf=settings["cnn_conf"],
    )


@app.get("/system-status")
def system_status():
    with JOBS_LOCK:
        active_count, queued_count = collect_system_counts_locked()
        active_by_mode = count_active_jobs_by_mode_locked()
    return jsonify(
        {
            "active_count": active_count,
            "queued_count": queued_count,
            "active_standard_count": active_by_mode["standard"],
            "active_rembg_count": active_by_mode["rembg"],
            "max_total_concurrent": MAX_TOTAL_CONCURRENT_JOBS,
            "max_standard_concurrent": MODE_CONCURRENCY_LIMITS["standard"],
            "max_rembg_concurrent": MODE_CONCURRENCY_LIMITS["rembg"],
        }
    )


@app.post("/api/focus-preview-jobs")
def create_focus_preview_job():
    upload = request.files.get("movie_file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "ピント確認用の動画ファイルを選択してください。"}), 400

    preview_id = uuid.uuid4().hex[:8]
    upload_name = sanitize_filename(upload.filename)
    upload_path = TMP_DIR / f"focus-source-{preview_id}-{upload_name}"
    preview_path = TMP_DIR / f"focus-preview-{preview_id}.png"
    upload.save(upload_path)
    settings = load_settings()
    workspace_dir = create_model_workspace(f"focus-preview-{preview_id}", settings)

    with PREVIEW_JOBS_LOCK:
        PREVIEW_JOBS[preview_id] = {
            "preview_id": preview_id,
            "file_name": upload_name,
            "status": "queued",
            "phase": "参照画像の生成待ちです。",
            "progress_percent": 0,
            "progress_text": "",
            "logs": deque(maxlen=200),
            "error_message": "",
            "pid": None,
            "upload_path": upload_path,
            "preview_path": preview_path,
            "workspace_dir": workspace_dir,
        }
        payload = build_preview_payload_locked(PREVIEW_JOBS[preview_id])

    worker = threading.Thread(target=execute_preview_job, args=(preview_id,), daemon=True)
    worker.start()
    return jsonify(payload), 201


@app.post("/api/model-check-jobs")
def create_model_check_job():
    upload = request.files.get("movie_file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "確認用の動画ファイルを選択してください。"}), 400

    pint_raw = request.form.get("pint_threshold", "").strip()
    if not pint_raw:
        return jsonify({"error": "確認用のピント閾値を指定してください。"}), 400
    try:
        pint_threshold = int(pint_raw)
    except ValueError:
        return jsonify({"error": "確認用のピント閾値は整数で指定してください。"}), 400
    if pint_threshold < 1:
        return jsonify({"error": "確認用のピント閾値は 1 以上を指定してください。"}), 400

    settings = load_settings()
    model_choices = build_model_choices()
    yolo_label_map = {item["value"]: item["label"] for item in model_choices["yolo"]}
    cnn_label_map = {item["value"]: item["label"] for item in model_choices["cnn"]}

    check_id = uuid.uuid4().hex[:10]
    upload_name = sanitize_filename(upload.filename)
    upload_path = TMP_DIR / f"model-check-source-{check_id}-{upload_name}"
    video_path = TMP_DIR / f"model-check-{check_id}.mp4"
    workspace_dir = create_model_workspace(f"model-check-{check_id}", settings)
    upload.save(upload_path)
    safe_unlink(video_path)

    with MODEL_CHECK_JOBS_LOCK:
        MODEL_CHECK_JOBS[check_id] = {
            "check_id": check_id,
            "file_name": upload_name,
            "status": "queued",
            "phase": "確認動画の生成待ちです。",
            "progress_percent": 0,
            "progress_text": "",
            "logs": deque(maxlen=300),
            "error_message": "",
            "pid": None,
            "upload_path": upload_path,
            "video_path": video_path,
            "workspace_dir": workspace_dir,
            "selected_yolo_model_label": yolo_label_map.get(settings["yolo_model"], settings["yolo_model"]),
            "selected_cnn_model_label": cnn_label_map.get(settings["cnn_model"], settings["cnn_model"]),
            "yolo_conf": settings["yolo_conf"],
            "cnn_conf": settings["cnn_conf"],
            "pint_threshold": pint_threshold,
        }
        payload = build_model_check_payload_locked(MODEL_CHECK_JOBS[check_id])

    worker = threading.Thread(target=execute_model_check_job, args=(check_id,), daemon=True)
    worker.start()
    return jsonify(payload), 201


@app.get("/api/model-check-jobs/<check_id>")
def model_check_status(check_id: str):
    with MODEL_CHECK_JOBS_LOCK:
        job = MODEL_CHECK_JOBS.get(check_id)
        if job is None:
            abort(404)
        payload = build_model_check_payload_locked(job)
    return jsonify(payload)


@app.get("/api/model-check-jobs/<check_id>/video")
def model_check_video(check_id: str):
    with MODEL_CHECK_JOBS_LOCK:
        job = MODEL_CHECK_JOBS.get(check_id)
        if job is None:
            abort(404)
        if job["status"] != "completed":
            return jsonify({"error": "確認動画はまだ準備できていません。"}), 409
        video_path = job["video_path"]
        if not video_path.exists():
            return jsonify({"error": "確認動画ファイルが見つかりません。"}), 404

    return send_file(video_path, mimetype="video/mp4")


@app.get("/api/model-check-jobs/<check_id>/download")
def model_check_download(check_id: str):
    with MODEL_CHECK_JOBS_LOCK:
        job = MODEL_CHECK_JOBS.get(check_id)
        if job is None:
            abort(404)
        if job["status"] != "completed":
            return jsonify({"error": "確認動画はまだ準備できていません。"}), 409
        video_path = job["video_path"]
        if not video_path.exists():
            return jsonify({"error": "確認動画ファイルが見つかりません。"}), 404
        download_name = f"{Path(job['file_name']).stem}-model-check.mp4"

    @after_this_request
    def remove_model_check_video(response):
        cleanup_model_check_job(check_id, terminate_running=False)
        return response

    return send_file(video_path, mimetype="video/mp4", as_attachment=True, download_name=download_name)


@app.post("/api/model-check-jobs/<check_id>/cleanup")
def model_check_cleanup(check_id: str):
    removed = cleanup_model_check_job(check_id, terminate_running=True)
    return jsonify({"removed": removed})


@app.get("/api/focus-preview-jobs/<preview_id>")
def focus_preview_status(preview_id: str):
    with PREVIEW_JOBS_LOCK:
        job = PREVIEW_JOBS.get(preview_id)
        if job is None:
            abort(404)
        payload = build_preview_payload_locked(job)
    return jsonify(payload)


@app.get("/api/focus-preview-jobs/<preview_id>/image")
def focus_preview_image(preview_id: str):
    with PREVIEW_JOBS_LOCK:
        job = PREVIEW_JOBS.get(preview_id)
        if job is None:
            abort(404)
        if job["status"] != "completed":
            return jsonify({"error": "参照画像はまだ準備できていません。"}), 409
        preview_path = job["preview_path"]
        if not preview_path.exists():
            return jsonify({"error": "参照画像ファイルが見つかりません。"}), 404

    @after_this_request
    def remove_preview_files(response):
        safe_unlink(preview_path)
        with PREVIEW_JOBS_LOCK:
            PREVIEW_JOBS.pop(preview_id, None)
        return response

    return send_file(preview_path, mimetype="image/png")


@app.post("/api/jobs")
def create_job():
    validated, error_info = validate_request_form()
    if error_info is not None:
        message, status_code = error_info
        return jsonify({"error": message}), status_code

    upload = validated["upload"]
    settings = load_settings()
    model_choices = build_model_choices()
    yolo_label_map = {item["value"]: item["label"] for item in model_choices["yolo"]}
    cnn_label_map = {item["value"]: item["label"] for item in model_choices["cnn"]}
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    original_name = sanitize_filename(upload.filename)
    upload_path = UPLOAD_DIR / f"{job_id}-{original_name}"
    upload.save(upload_path)
    workspace_dir = create_model_workspace(job_id, settings)

    script_path = workspace_dir / "muscut.py"
    output_subdir = "selected_imgs"
    if validated["processing_mode"] == "rembg":
        script_path = workspace_dir / "muscut_with_rembg.py"
        output_subdir = "with_rembg"

    output_stem = upload_path.stem
    output_root = workspace_dir / "croped_image" / output_stem
    output_dir = output_root / output_subdir
    if validated["processing_mode"] == "rembg" and validated["dev_flag"]:
        output_dir = output_root
    zip_path = DOWNLOAD_DIR / f"{job_id}.zip"

    safe_rmtree(output_root)
    safe_unlink(zip_path)

    command = build_command(
        script_path=script_path,
        upload_path=upload_path,
        image_format=validated["image_format"],
        extract_count=validated["extract_count"],
        cnn_conf=float(settings["cnn_conf"]),
        pint_threshold=validated["pint_threshold"],
        all_extract=validated["all_extract"],
        tool=validated["tool"],
        without_cnn=validated["without_cnn"],
        dev_flag=validated["dev_flag"],
    )

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "file_name": original_name,
            "status": "queued",
            "phase": "順番待ちです。",
            "progress_percent": 0,
            "progress_text": "",
            "processing_mode": validated["processing_mode"],
            "logs": deque(maxlen=200),
            "download_name": None,
            "image_count": 0,
            "command_list": command,
            "error_message": "",
            "pid": None,
            "queue_position": 0,
            "output_dir": output_dir,
            "output_root": output_root,
            "zip_path": zip_path,
            "upload_path": upload_path,
            "workspace_dir": workspace_dir,
            "selected_yolo_model": settings["yolo_model"],
            "selected_cnn_model": settings["cnn_model"],
            "selected_yolo_model_label": yolo_label_map.get(settings["yolo_model"], settings["yolo_model"]),
            "selected_cnn_model_label": cnn_label_map.get(settings["cnn_model"], settings["cnn_model"]),
            "yolo_conf": settings["yolo_conf"],
            "cnn_conf": settings["cnn_conf"],
            "pint_threshold": validated["pint_threshold"] if validated["pint_threshold"] is not None else 2600,
        }
        JOB_QUEUE.append(job_id)
        refresh_queue_positions_locked()

    schedule_jobs()

    with JOBS_LOCK:
        payload = build_job_payload_locked(JOBS[job_id])

    return jsonify(payload), 201


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        jobs = [build_job_payload_locked(job) for job in JOBS.values()]
    jobs.sort(key=lambda item: item["job_id"], reverse=True)
    return jsonify({"jobs": jobs})


@app.post("/api/jobs/clear")
def clear_jobs():
    removed = 0
    with JOBS_LOCK:
        removable_ids = [
            job_id
            for job_id, job in JOBS.items()
            if job["status"] in {"completed", "error"}
        ]
        for job_id in removable_ids:
            JOBS.pop(job_id, None)
            removed += 1
    return jsonify({"removed": removed})


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        if job["status"] not in {"completed", "error"}:
            return jsonify({"error": "実行中または待機中のジョブは削除できません。"}), 409
        JOBS.pop(job_id, None)
    return jsonify({"deleted": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        payload = build_job_payload_locked(job)
    return jsonify(payload)


@app.get("/api/download-bulk/status")
def bulk_download_status():
    with JOBS_LOCK:
        available_count = len(collect_bulk_download_items_locked())
    return jsonify(
        {
            "available_count": available_count,
            "download_url": url_for("download_bulk_result") if available_count else None,
        }
    )


@app.get("/download/<path:filename>")
def download_result(filename: str):
    file_path = (DOWNLOAD_DIR / filename).resolve()
    if not str(file_path).startswith(str(DOWNLOAD_DIR.resolve())):
        abort(404)
    if not file_path.exists():
        abort(404)

    @after_this_request
    def remove_download_file(response):
        safe_unlink(file_path)
        mark_download_consumed(filename)
        return response

    return send_file(file_path, as_attachment=True)


@app.get("/download-bulk")
def download_bulk_result():
    bundle_path, item_count, zip_paths, download_names = build_bulk_download_zip()
    if bundle_path is None or item_count == 0:
        return jsonify({"error": "まとめてダウンロードできる完了ジョブがありません。"}), 404

    @after_this_request
    def remove_bundle_file(response):
        safe_unlink(bundle_path)
        for zip_path in zip_paths:
            safe_unlink(zip_path)
        mark_multiple_downloads_consumed(download_names)
        return response

    return send_file(bundle_path, as_attachment=True, download_name=bundle_path.name)


if __name__ == "__main__":
    host = os.environ.get("MUSCUT_WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("MUSCUT_WEBUI_PORT", "8000"))
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host

    def open_browser():
        webbrowser.open(f"http://{browser_host}:{port}")

    threading.Timer(1.0, open_browser).start()
    app.run(host=host, port=port, debug=False)
