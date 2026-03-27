import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from focus_threshold_checker import YOLO, main as checker_main, os_name, plt, tf


def generate_focus_preview(
    movie_path: Path,
    output_path: Path,
    num_images: int = 10,
    batch_size: int = 8,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if os_name == "Darwin":
        model = YOLO("muscut_models/yolo.mlmodel", task="detect")
    else:
        model = YOLO("muscut_models/yolo.pt")

    cnn_model = tf.keras.models.load_model("muscut_models/cnn/savedmodel")
    original_show = plt.show

    def save_show(*_args, **_kwargs):
        plt.savefig(output_path, bbox_inches="tight", dpi=144)

    try:
        plt.show = save_show
        checker_main(str(movie_path), model, cnn_model, num_images=num_images, b_size=batch_size)
    finally:
        plt.show = original_show
        plt.close("all")
        del cnn_model
        del model
        gc.collect()
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    if not output_path.exists():
        raise RuntimeError("Focus preview image was not generated.")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate focus preview image for WebUI.")
    parser.add_argument("--movie", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    generate_focus_preview(
        Path(args.movie),
        Path(args.output),
        num_images=args.num_images,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
