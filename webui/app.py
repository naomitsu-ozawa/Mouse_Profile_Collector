import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, after_this_request, jsonify, redirect, render_template, request, send_file, url_for


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
UPLOAD_DIR = APP_DIR / "uploads"
DOWNLOAD_DIR = APP_DIR / "downloads"
TMP_DIR = APP_DIR / "tmp"
MAX_CONCURRENT_JOBS = 3

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
        "muscut.py",
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
    if "All Done!" in line or "処理が完了しました" in line:
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


def build_job_payload_locked(job: dict) -> dict:
    active_count, queued_count = collect_system_counts_locked()
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "phase": job["phase"],
        "progress_percent": job["progress_percent"],
        "progress_text": job["progress_text"],
        "image_count": job["image_count"],
        "command": job["command"],
        "error_message": job["error_message"],
        "logs": list(job["logs"]),
        "queue_position": job["queue_position"],
        "active_count": active_count,
        "queued_count": queued_count,
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "download_url": (
            url_for("download_result", filename=job["download_name"])
            if job["download_name"]
            else None
        ),
    }


def schedule_jobs() -> None:
    to_start: list[str] = []
    with JOBS_LOCK:
        while len(ACTIVE_JOB_IDS) < MAX_CONCURRENT_JOBS and JOB_QUEUE:
            job_id = JOB_QUEUE.popleft()
            job = JOBS.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            ACTIVE_JOB_IDS.add(job_id)
            job["status"] = "running"
            job["queue_position"] = 0
            job["phase"] = "処理を開始しました。"
            to_start.append(job_id)
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
        return None, (
            render_template(
                "error.html",
                error_title="動画ファイルが未指定です",
                error_message="動画ファイルを選択してから実行してください。",
            ),
            400,
        )

    image_format = request.form.get("image_format", "png").lower()
    if image_format not in {"png", "jpg"}:
        return None, (
            render_template(
                "error.html",
                error_title="画像形式が不正です",
                error_message="画像形式は png または jpg を指定してください。",
            ),
            400,
        )

    tool = request.form.get("tool", "default").strip() or "default"
    if tool not in {"default", "kmeans_image_extractor"}:
        return None, (
            render_template(
                "error.html",
                error_title="ツール指定が不正です",
                error_message="選択したツールは WebUI では利用できません。",
            ),
            400,
        )

    all_extract = request.form.get("all_extract") == "on"
    without_cnn = request.form.get("without_cnn") == "on"
    dev_flag = request.form.get("dev_flag") == "on"

    extract_count = None
    if not all_extract:
        try:
            extract_count = int(request.form.get("extract_count", "").strip())
        except ValueError:
            return None, (
                render_template(
                    "error.html",
                    error_title="抽出枚数が不正です",
                    error_message="抽出枚数には整数を指定してください。",
                ),
                400,
            )
        if extract_count < 1:
            return None, (
                render_template(
                    "error.html",
                    error_title="抽出枚数が不正です",
                    error_message="抽出枚数は 1 以上を指定してください。",
                ),
                400,
            )

    pint_threshold = None
    pint_raw = request.form.get("pint_threshold", "").strip()
    if pint_raw:
        try:
            pint_threshold = int(pint_raw)
        except ValueError:
            return None, (
                render_template(
                    "error.html",
                    error_title="ピント閾値が不正です",
                    error_message="ピント閾値には整数を指定してください。",
                ),
                400,
            )
        if pint_threshold < 1:
            return None, (
                render_template(
                    "error.html",
                    error_title="ピント閾値が不正です",
                    error_message="ピント閾値は 1 以上を指定してください。",
                ),
                400,
            )

    return {
        "upload": upload,
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
    return render_template("index.html")


@app.get("/system-status")
def system_status():
    with JOBS_LOCK:
        active_count, queued_count = collect_system_counts_locked()
    return jsonify(
        {
            "active_count": active_count,
            "queued_count": queued_count,
            "max_concurrent": MAX_CONCURRENT_JOBS,
        }
    )


@app.post("/run")
def run_muscut():
    validated, error_response = validate_request_form()
    if error_response is not None:
        return error_response

    upload = validated["upload"]
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    upload_name = sanitize_filename(upload.filename)
    upload_path = UPLOAD_DIR / f"{job_id}-{upload_name}"
    upload.save(upload_path)

    output_stem = upload_path.stem
    output_root = REPO_ROOT / "croped_image" / output_stem
    output_dir = REPO_ROOT / "croped_image" / output_stem / "selected_imgs"
    zip_path = DOWNLOAD_DIR / f"{job_id}.zip"

    if output_dir.exists():
        shutil.rmtree(output_dir.parent)
    if zip_path.exists():
        zip_path.unlink()

    command = build_command(
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
            "status": "queued",
            "phase": "順番待ちです。",
            "progress_percent": 0,
            "progress_text": "",
            "logs": deque(maxlen=200),
            "download_name": None,
            "image_count": 0,
            "command": " ".join(command),
            "command_list": command,
            "error_message": "",
            "pid": None,
            "queue_position": 0,
            "output_dir": output_dir,
            "output_root": output_root,
            "zip_path": zip_path,
            "upload_path": upload_path,
        }

        if len(ACTIVE_JOB_IDS) < MAX_CONCURRENT_JOBS and not JOB_QUEUE:
            ACTIVE_JOB_IDS.add(job_id)
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["phase"] = "処理を開始しました。"
            start_now = True
        else:
            JOB_QUEUE.append(job_id)
            refresh_queue_positions_locked()

    if start_now:
        worker = threading.Thread(target=execute_job, args=(job_id,), daemon=True)
        worker.start()

    return redirect(url_for("job_page", job_id=job_id))


@app.get("/jobs/<job_id>")
def job_page(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        initial = build_job_payload_locked(job)
    return render_template("job.html", initial=initial)


@app.get("/jobs/<job_id>/status")
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
        return response

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    host = os.environ.get("MUSCUT_WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("MUSCUT_WEBUI_PORT", "8000"))
    app.run(host=host, port=port, debug=False)
