from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _collect_nvidia_lib_dirs(project_root: Path) -> list[str]:
    """Collect NVIDIA CUDA library directories inside the uv virtual environment."""
    site_packages = (
        project_root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_root = site_packages / "nvidia"

    if not nvidia_root.exists():
        return []

    return sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "webui" / "app.py"

    env = os.environ.copy()

    if platform.system() == "Linux":
        nvidia_lib_dirs = _collect_nvidia_lib_dirs(project_root)

        if nvidia_lib_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            cuda_paths = ":".join(nvidia_lib_dirs)
            env["LD_LIBRARY_PATH"] = (
                f"{cuda_paths}:{existing}" if existing else cuda_paths
            )

    os.execvpe(
        sys.executable,
        [sys.executable, str(app_path)],
        env,
    )
