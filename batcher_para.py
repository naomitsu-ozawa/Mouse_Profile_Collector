import os

import argparse

import gc

# import time
import cProfile
import functools
import glob
import shutil
import subprocess
import threading
import pstats
import io
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from time import sleep

import torch

import getch
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

"""_summary_
batcher_single.pyを並列実行する
├── 00_male　←ここを指定する
│   ├── animal01　←このフォルダごとで並列化
│   ├── animal02
│   ├── animal03
│   ├── animal04
│   └── animal05

"""


def get_parallel_processing_limit():
    """
    プライマリGPUのメモリ情報を基に、並列処理の上限（2400MB単位）を求める関数。
    システムが使用しているであろう1GBを差し引いて計算します。

    Returns:
        tuple: (total_memory_MB, free_memory_MB, used_memory_MB, parallel_limit)
            total_memory_MB (float): GPUの総メモリ（MB単位）
            free_memory_MB (float): GPUの空きメモリ（MB単位）
            used_memory_MB (float): GPUの使用中メモリ（MB単位）
            parallel_limit (int): 並列処理可能なプロセスの最大数（2400MBごと）
        None: GPUが利用できない場合
    """
    if not torch.cuda.is_available():
        print("利用可能なGPUがありません。")
        return None

    try:
        # プライマリGPU（通常はGPU 0）の空きメモリと総メモリを取得
        free_memory, total_memory = torch.cuda.mem_get_info(torch.device("cuda:0"))

        # 総メモリをMB単位に変換
        total_memory_MB = total_memory / 1024**2
        # print(total_memory_MB)
        # 空きメモリをMB単位に変換
        free_memory_MB = free_memory / 1024**2
        # print(free_memory_MB)

        usage_memory_MB = total_memory_MB - free_memory_MB
        # print(usage_memory_MB)

        # システム使用分として1GB（1024MB）を差し引く
        usable_memory_MB = free_memory_MB - 1024

        # 使用可能メモリを1200MB単位で割り、小数点以下を切り捨て
        parallel_limit = max(
            0, int(usable_memory_MB // 1200)
        )  # 0未満にならないように制限

        return total_memory_MB, free_memory_MB, usage_memory_MB, parallel_limit
    except RuntimeError as e:
        print(f"エラー: {e}")
        return None
    finally:
        # メモリ解放
        del free_memory, total_memory
        gc.collect()  # ガベージコレクタを実行
        torch.cuda.empty_cache()  # GPUのキャッシュを解放


# パスワード入力のマスク処理
def masked_input(prompt=""):
    print(prompt, end="", flush=True)
    password = ""
    while True:
        char = getch.getch()
        if char == "\n" or char == "\r":  # Enterキーを検知
            break
        elif char == "\x08" or char == "\x7f":  # バックスペースを検知
            if len(password) > 0:
                # カーソルを戻してスペースで消して再度カーソルを戻す
                print("\b \b", end="", flush=True)
                password = password[:-1]
        else:
            print("*", end="", flush=True)
            password += char
    print()  # プロンプトを次の行に移動
    return password


# OK
def run_batch_predictions(
    sub_dir, pint, num_items, progress_dict, worker_id, x, wc_flag, all_flag
):
    console = Console()
    spinner = Spinner(
        "dots", text=f"[bold green]Processing {os.path.basename(sub_dir)}..."
    )
    progress_dict[worker_id] = (spinner, sub_dir)
    # console.log(spinner.text)  # スピナーの初期テキストをログに出力

    command = [
        "python",
        "batcher_single.py",
        "-f",
        sub_dir,
        "-p",
        str(pint),
        "-n",
        str(num_items),
        "-wc" if wc_flag else "",
        "-a" if all_flag else "",
    ]

    # サブプロセスを実行し、標準出力と標準エラーをキャプチャする
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    def stream_output(stream, is_error=False):
        for line in iter(stream.readline, ""):
            line = line.strip()
            if not is_error:
                # 標準出力の行は無視してスピナーのテキストを更新
                spinner.text = f"{os.path.basename(sub_dir)}: {line}"

        stream.close()

    # 標準出力と標準エラーを非同期に読み取るスレッドを開始する
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, True))

    stdout_thread.start()
    stderr_thread.start()

    # サブプロセスが終了するのを待つ
    process.wait()

    # スレッドが終了するのを待つ
    stdout_thread.join()
    stderr_thread.join()

    # 完了したらスピナーを更新
    spinner.text = f"Finished {os.path.basename(sub_dir)}"
    # console.log(f"Finished {os.path.basename(sub_dir)}")

    return x + 1


# サブディレクトリを取得する関数を定義
def get_subdirectories(root_dir):
    subdirectories = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    return subdirectories


# k-means-tempフォルダの画像を無視する
def get_all_image_files(directory):
    # 対応する画像ファイルの拡張子を指定
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    # 除外するフォルダ名のリスト
    exclude_folders = {"k-means_temp", "selected_imgs"}

    # サブフォルダも含めて画像ファイルを取得
    image_files = []
    for root, _, files in os.walk(directory):
        if not any(exclude in root for exclude in exclude_folders):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

    return image_files


# ファイル移動
def move_files(files, destination):
    # 移動先フォルダが存在しない場合は作成
    if not os.path.exists(destination):
        os.makedirs(destination)
    for file in files:
        shutil.move(file, destination)


# croped imgsフォルダを削除
def delete_croped_imgs_folders(directory):
    # 指定したフォルダ内のすべてのフォルダを再帰的に検索
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "croped_imgs":
                folder_to_delete = os.path.join(root, dir_name)
                shutil.rmtree(folder_to_delete)
                # print(f"{folder_to_delete} フォルダとその配下を削除しました。")


# 進捗表示を行う関数
def update_progress(num_workers, progress_dict):
    console = Console()
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            table = Table()
            table.add_column("Worker ID")
            table.add_column("Status")
            for worker_id in range(num_workers):
                spinner, full_sub_dir = progress_dict[
                    worker_id
                ]  # スピナーとサブディレクトリのフルパスを取得
                table.add_row(f"Worker {worker_id + 1}", spinner)
            live.update(Panel(table, title="Worker Progress"))
            sleep(0.1)


# メイン関数
def main(folder_path, num_workers, pint, num_items, wc_flag, all_flag):

    max_attempts = 3
    attempts = 0

    if su_pass:
        while attempts < max_attempts:
            pass_input = masked_input("Please enter sudo password:")
            check_command = f"echo {pass_input} | sudo -S echo pass_check_ok"
            check_result = subprocess.run(
                check_command, shell=True, capture_output=True, text=True
            )
            attempts += 1
            if check_result.returncode == 0:
                print("sudo check ok")
                print("Output:", check_result.stdout)
                break
            else:
                print("password failed")
                # print("Error:", check_result.stderr)
                if attempts < max_attempts:
                    print("sudo パスワードが間違っています。もう一度試してください。")
                else:
                    return print("最大試行回数に達しました。終了します。")

    # サブディレクトリのリストを取得
    subdirectories = get_subdirectories(folder_path)
    # subdirectories.sort()

    # 進捗情報を格納する辞書
    progress_dict = {i: ("", "") for i in range(num_workers)}

    # 進捗更新スレッドを開始
    progress_thread = threading.Thread(
        target=update_progress, args=(num_workers, progress_dict), daemon=True
    )
    progress_thread.start()

    # スレッドプールを使用してサブディレクトリを処理
    results = []
    x = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                run_batch_predictions,
                sub_dir,
                pint,
                num_items,
                progress_dict,
                i % num_workers,
                x,
                wc_flag,
                all_flag,
            )
            for i, sub_dir in enumerate(subdirectories)
        ]

        for future in as_completed(futures):
            result = future.result()
            x += result
            results.append(result)

        # (done, notdone) = wait(futures)
        # executor.shutdown(wait=True)

    # プログレススレッドが出力を完了するのを待つ
    sleep(2)

    # Richコンソールでカーソルを表示
    console = Console()
    console.show_cursor()

    sleep(1)

    # print(results)

    destination = f"{folder_path}/extract_images"

    images = get_all_image_files(folder_path)
    move_files(images, destination)

    delete_croped_imgs_folders(folder_path)

    if su_pass:
        command = "sh clear_mem_cache.sh"
        result = subprocess.run(
            ["sudo", "-S"] + command.split(),
            input=str(pass_input) + "\n",
            text=True,
            capture_output=True,
        )
        print(result.stdout)
        print(result.stderr)


def get_args():
    parser = argparse.ArgumentParser(
        prog="muscut_batcher",  # プログラム名
        # usage="",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # opiton file name
    parser.add_argument(
        "-f",
        "--folder_path",
        help="解析したい動画のあるフォルダのパスを指定してください。",
        # type=str
        # required=True,
    )

    # option pint check
    parser.add_argument(
        "-p",
        "--pint",
        default=2600,
        type=int,
        help="ピントチェックの閾値、デフォルト2600",
    )

    # option pint check
    parser.add_argument(
        "-n",
        "--num_items",
        type=int,
        help="抽出枚数",
        # required=True,
    )

    # opiton memoru purge pass
    parser.add_argument(
        "-ps",
        "--su_pass",
        help="バッチ処理中にメモリを開放したいときにSudo権限で開放するため",
        action="store_true",
        # required=True,
    )

    # option worker num
    parser.add_argument(
        "-nw",
        "--num_workers",
        default=2,
        type=int,
        help="並列度　１以上の整数　多くするとGPUメモリーを使い切るので注意",
    )

    # option preview show
    parser.add_argument(
        "-wc",
        "--without_cnn",
        action="store_true",
        help="CNNモデルによる分類をスキップします。",
    )

    # option all extract
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="all_extract",
    )

    args_list = parser.parse_args()

    return args_list


if __name__ == "__main__":

    args = get_args()

    folder_path = args.folder_path
    pint = args.pint
    num_items = args.num_items
    su_pass = args.su_pass
    num_workers = args.num_workers
    wc_flag = args.without_cnn
    all_flag = args.all

    if num_items is None or all_flag is None:
        print("抽出枚数を指定してください。もしくは--allオプションを指定してください。")
        exit()

    check_result = get_parallel_processing_limit()
    if check_result is not None:
        total_memory_MB, free_memory_MB, usage_memory_MB, parallel_limit = check_result
        print(f"GPUの総メモリ: {total_memory_MB:.2f} MB")
        print(f"GPUの空きメモリ: {free_memory_MB:.2f} MB")
        print(f"推奨される最大並列処理数: {parallel_limit} workers")
        if num_workers > parallel_limit:
            print(
                f"警告：指定された処理数（{num_workers}）は推奨される最大並列処理数（{parallel_limit}）を超えています。"
            )
            print(
                f"指定された処理数（{num_workers}）を推奨される処理数（{parallel_limit}）へ制限します。"
            )
            num_workers = parallel_limit
        elif num_workers == parallel_limit:
            print(f"指定された処理数（{num_workers}）は適切に設定されています。")
        elif num_workers < parallel_limit:
            print(
                f"指定された処理数（{num_workers}）は推奨される最大並列処理数（{parallel_limit}）を下回っています。"
            )

    total_cpu_cores = os.cpu_count()
    threads_per_worker = total_cpu_cores // num_workers
    # 環境変数を設定
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

    def wrapped_main():
        main(folder_path, num_workers, pint, num_items, wc_flag, all_flag)

    # cProfileのプロファイラを作成
    pr = cProfile.Profile()
    pr.enable()  # プロファイリングを開始
    wrapped_main()  # プロファイル対象の関数を実行
    pr.disable()  # プロファイリングを終了

    # 結果を解析して出力
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats("main")
    # 必要な行を厳密に抽出して表示
    profile_output = s.getvalue()
    for line in profile_output.split("\n"):
        if "function calls" in line:
            print(line)

    # print(s.getvalue())
