from __future__ import annotations

import os
import threading
import webbrowser

try:
    from . import app_core as _core
    from .gpu_selection import install as _install_gpu_selection
except ImportError:
    import app_core as _core
    from gpu_selection import install as _install_gpu_selection


_install_gpu_selection(_core)
app = _core.app


def __getattr__(name: str):
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


def main() -> None:
    host = os.environ.get("MUSCUT_WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("MUSCUT_WEBUI_PORT", "8000"))
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host

    def open_browser() -> None:
        webbrowser.open(f"http://{browser_host}:{port}")

    threading.Timer(1.0, open_browser).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
