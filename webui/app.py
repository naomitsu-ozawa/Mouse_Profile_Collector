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

from flask import Flask, abort, after_this_request, jsonify, render_template, request, send_file, url_for


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
UPLOAD_DIR = APP_DIR / "uploads"
DOWNLOAD_DIR = APP_DIR / "downloads"
TMP_DIR = APP_DIR / "tmp"
MAX_TOTAL_CONCURRENT_JOBS = 8
MODE_CONCURRENCY_LIMITS = {
    "standard": 8,
    "rembg": 3,
}

for directory in (UPLOAD_DIR, DOWNLOAD_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ACTIVE_JOB_IDS: set[str] = set()
JOB_QUEUE: deque[str] = deque()
PROGRESS_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
PROGRESS_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)")


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe or "upload.mp4"


def build_command(
    script_name: str,
    upload_path: Path,
    image_format: str,
    extract_count: int | None,
    pint_threshold: int | None,
    all_extract: bool,
    tool: str,
    without_cnn: bool,
    dev_flag: bool,
) -> list[str]:
    command = [
        sys.executable,
        script_name,
        "-f",
        str(upload_path),
        "-i",
        image_format,
    ]
    if extract_count is not None:
        command.extend(["-n", str(extract_count)])
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


def build_bulk_download_zip() -> tuple[Path | None, int, list[Path], list[str]]:
    completed_items: list[tuple[str, Path, str]] = []
    with JOBS_LOCK:
        for job in JOBS.values():
            if job["status"] != "completed":
                continue
            download_name = job.get("download_name")
            if not download_name:
                continue
            zip_path = DOWNLOAD_DIR / download_name
            if zip_path.exists():
                completed_items.append((job["file_name"], zip_path, download_name))

    if not completed_items:
        return None, 0, [], []

    bundle_name = f"bulk-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.zip"
    bundle_path = TMP_DIR / bundle_name
    safe_unlink(bundle_path)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, item_path, _download_name in completed_items:
            archive.write(item_path, arcname=f"{Path(file_name).stem}.zip")

    zip_paths = [item_path for _file_name, item_path, _download_name in completed_items]
    filenames = [download_name for _file_name, _item_path, download_name in completed_items]
    return bundle_path, len(completed_items), zip_paths, filenames


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
        output_dir = job["output_dir"]
        output_root = job["output_root"]
        zip_path = job["zip_path"]
        upload_path = job["upload_path"]

    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
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
        image_count = len(list(output_dir.glob("*")))
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
    return render_template(
        "index.html",
        max_total_concurrent=MAX_TOTAL_CONCURRENT_JOBS,
        max_standard_concurrent=MODE_CONCURRENCY_LIMITS["standard"],
        max_rembg_concurrent=MODE_CONCURRENCY_LIMITS["rembg"],
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


@app.post("/api/jobs")
def create_job():
    validated, error_info = validate_request_form()
    if error_info is not None:
        message, status_code = error_info
        return jsonify({"error": message}), status_code

    upload = validated["upload"]
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    original_name = sanitize_filename(upload.filename)
    upload_path = UPLOAD_DIR / f"{job_id}-{original_name}"
    upload.save(upload_path)

    script_name = "muscut.py"
    output_subdir = "selected_imgs"
    if validated["processing_mode"] == "rembg":
        script_name = "muscut_with_rembg.py"
        output_subdir = "with_rembg"

    output_stem = upload_path.stem
    output_root = REPO_ROOT / "croped_image" / output_stem
    output_dir = output_root / output_subdir
    zip_path = DOWNLOAD_DIR / f"{job_id}.zip"

    safe_rmtree(output_root)
    safe_unlink(zip_path)

    command = build_command(
        script_name=script_name,
        upload_path=upload_path,
        image_format=validated["image_format"],
        extract_count=validated["extract_count"],
        pint_threshold=validated["pint_threshold"],
        all_extract=validated["all_extract"],
        tool=validated["tool"],
        without_cnn=validated["without_cnn"],
        dev_flag=validated["dev_flag"],
    )

    start_now = False
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
        }

        if not JOB_QUEUE and can_start_job_locked(JOBS[job_id]):
            ACTIVE_JOB_IDS.add(job_id)
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["phase"] = "処理を開始しました。"
            start_now = True
        else:
            JOB_QUEUE.append(job_id)
            refresh_queue_positions_locked()

        payload = build_job_payload_locked(JOBS[job_id])

    if start_now:
        worker = threading.Thread(target=execute_job, args=(job_id,), daemon=True)
        worker.start()

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
