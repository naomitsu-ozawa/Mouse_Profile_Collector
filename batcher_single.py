import os
import glob
import argparse
import shutil
from tqdm import tqdm

import platform
import shlex


os_name = platform.system()


def main(folder_path, num_items, pint, wc_flag, all_flag):
    """_summary_
    指定したフォルダ内の動画を解析する
    ├──  animal01 ←　ここを指定する
    │   ├──  C0013.MP4
    │   ├──  C0014.MP4
    │   └──  C0015.MP4
    Args:
        folder_path (_type_): _description_
        num_items (_type_): _description_
        pint (_type_): _description_
    """
    # print(folder_path)
    video_files = find_videos_in_subdirectories(folder_path)
    # 動画ファイルの一覧を出力
    # print(f"debag video_files:{video_files}")

    video_list = []

    for video_file in video_files:
        video_file = shlex.quote(video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        video_list.append(video_name)
        # print(f"debag video_file:{video_name}")
        # print(f"debag video_file:{video_file}")

        # muscut　実行
        exe_python = f"python muscut.py -f {video_file} -n {num_items} -p {pint}"
        if wc_flag:
            exe_python += " -wc"
        if all_flag:
            exe_python += " -a"
        os.system(exe_python)

    subdir_name = os.path.basename(folder_path)
    destination_dir_name = f"{folder_path}/croped_imgs/"

    create_directory_if_not_exists(destination_dir_name)

    # 抽出した画像を移動
    # image_files = glob.glob(f"./croped_image/**/**/*.png")
    base_directory = "./croped_image/"
    image_files = get_images_from_folder_names(video_list, base_directory)
    # print(f"debag:{image_files}")
    move_image_files(image_files, destination_dir_name)

    # del_dirs = get_subdirectories("./croped_image")
    del_dirs = find_folders(video_list, base_directory)
    for del_dir in del_dirs:
        shutil.rmtree(del_dir)

    print("all done")

    return True


def move_image_files(image_files, destination_folder):
    pbar = tqdm(image_files)
    for file_path in pbar:
        try:
            shutil.move(file_path, destination_folder)
            # print(f"ファイル {file_path} を {destination_folder} に移動しました")
        except Exception as e:
            print(f"ファイル {file_path} の移動中にエラーが発生しました: {e}")


def create_directory_if_not_exists(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"ディレクトリ {directory} を作成しました")
        else:
            print(f"ディレクトリ {directory} はすでに存在します")
    except Exception as e:
        print(f"ディレクトリ {directory} の作成中にエラーが発生しました: {e}")


def get_subdirectories(root_dir):
    subdirectories = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    return subdirectories


def find_folders(folder_names, base_directory):
    matched_folders = []
    for root, dirs, files in os.walk(base_directory):
        for folder_name in folder_names:
            if folder_name in dirs:
                matched_folders.append(os.path.join(root, folder_name))
    return matched_folders


def find_videos_in_subdirectories(root_dir):
    video_extensions = ["MP4", "mp4", "avi", "mkv", "mov"]  # 動画ファイルの拡張子リスト
    videos = []  # 動画ファイルのリストを初期化

    for extension in video_extensions:
        search_pattern = (
            root_dir + "/**/*." + extension
        )  # ワイルドカードを使ってパスのパターンを作成
        videos.extend(
            glob.glob(search_pattern, recursive=True)
        )  # ワイルドカードを使って動画ファイルを検索し、リストに追加

    return videos


def get_images_from_folder_names(
    folder_list, base_directory, extensions=[".jpg", ".jpeg", ".png"]
):
    images = []
    for folder_name in folder_list:
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        images.append(os.path.join(root, file))
    return images


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

    # option extract number
    parser.add_argument(
        "-n",
        "--num_items",
        type=int,
        help="抽出枚数を指定してください",
    )

    # option pint check
    parser.add_argument(
        "-p",
        "--pint",
        type=int,
        help="ピントチェックの閾値、デフォルト2600",
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
    num_items = args.num_items
    pint = args.pint
    wc_flag = args.without_cnn
    all_flag = args.all

    main(folder_path, num_items, pint, wc_flag, all_flag)
    print("\033[32m処理が完了しました。\033[0m")


##############test##################

# folder_path = (
#     "/media/idm-kurume/ex_ssd/ex_ssd/project_MYZ/formjudge/sex_test/formjudge_test"
# )

# num_items = 30

# main(folder_path, num_items)

############ テスト用#################
