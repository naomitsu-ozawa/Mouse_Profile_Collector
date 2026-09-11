from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import ModuleType


GPU_AUTO = "auto"
GPU_MARKER_NAME = ".webui-gpu-selection"
_REAL_SUBPROCESS = subprocess


def list_nvidia_gpus(system_name: str) -> list[dict[str, str]]:
    if system_name == "Darwin":
        return []

    try:
        result = _REAL_SUBPROCESS.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, _REAL_SUBPROCESS.SubprocessError):
        return []

    gpus: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue

        index, name, gpu_uuid, memory_total = parts
        if not gpu_uuid:
            continue

        label = f"GPU {index} - {name}"
        if memory_total:
            label += f" - {memory_total} MiB"
        gpus.append({"value": gpu_uuid, "label": label})

    return gpus


def resolve_selected_gpu(
    settings: dict[str, str | float | int | bool],
    gpu_choices: list[dict[str, str]],
) -> str:
    selected_gpu = str(settings.get("selected_gpu", GPU_AUTO)).strip() or GPU_AUTO
    if selected_gpu == GPU_AUTO:
        return GPU_AUTO

    valid_values = {choice["value"] for choice in gpu_choices}
    return selected_gpu if selected_gpu in valid_values else GPU_AUTO


def build_child_environment(
    selected_gpu: str,
    inherited_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = inherited_env.copy() if inherited_env is not None else os.environ.copy()
    if selected_gpu != GPU_AUTO:
        env["CUDA_VISIBLE_DEVICES"] = selected_gpu
    return env


class _SubprocessProxy:
    def __getattr__(self, name: str):
        return getattr(_REAL_SUBPROCESS, name)

    def Popen(self, *args, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd is not None:
            marker_path = Path(cwd) / GPU_MARKER_NAME
            try:
                selected_gpu = marker_path.read_text(encoding="utf-8").strip()
            except OSError:
                selected_gpu = GPU_AUTO

            if selected_gpu and selected_gpu != GPU_AUTO:
                kwargs["env"] = build_child_environment(
                    selected_gpu,
                    kwargs.get("env"),
                )

        return _REAL_SUBPROCESS.Popen(*args, **kwargs)


def install(core: ModuleType) -> None:
    if getattr(core, "_GPU_SELECTION_INSTALLED", False):
        return
    core._GPU_SELECTION_INSTALLED = True

    core.DEFAULT_SETTINGS["selected_gpu"] = GPU_AUTO

    original_ensure_valid_settings = core.ensure_valid_settings

    def ensure_valid_settings_with_gpu(selected: dict[str, str | bool]):
        validated, error_message = original_ensure_valid_settings(selected)
        if validated is None:
            return None, error_message

        selected_gpu = str(core.request.form.get("selected_gpu", GPU_AUTO)).strip() or GPU_AUTO
        gpu_choices = list_nvidia_gpus(core.SYSTEM_NAME)
        valid_values = {GPU_AUTO}
        valid_values.update(choice["value"] for choice in gpu_choices)
        if selected_gpu not in valid_values:
            return None, "使用GPUの選択が不正です。GPU一覧を再確認してください。"

        validated["selected_gpu"] = selected_gpu
        return validated, None

    core.ensure_valid_settings = ensure_valid_settings_with_gpu

    @core.app.context_processor
    def inject_gpu_settings():
        if core.request.endpoint != "settings_page":
            return {}

        gpu_choices = list_nvidia_gpus(core.SYSTEM_NAME)
        settings = core.load_settings()
        return {
            "gpu_choices": gpu_choices,
            "selected_gpu": resolve_selected_gpu(settings, gpu_choices),
        }

    original_create_model_workspace = core.create_model_workspace

    def create_model_workspace_with_gpu(
        job_token: str,
        settings: dict[str, str | float | int | bool],
    ) -> Path:
        workspace = original_create_model_workspace(job_token, settings)
        gpu_choices = list_nvidia_gpus(core.SYSTEM_NAME)
        selected_gpu = resolve_selected_gpu(settings, gpu_choices)
        (workspace / GPU_MARKER_NAME).write_text(selected_gpu, encoding="utf-8")
        return workspace

    core.create_model_workspace = create_model_workspace_with_gpu
    core.subprocess = _SubprocessProxy()
