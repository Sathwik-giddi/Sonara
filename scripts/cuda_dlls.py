"""Make venv-installed CUDA runtime DLLs loadable on Windows.

ctranslate2 (faster-whisper's backend) links cuBLAS and cuDNN at runtime. The
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels drop those DLLs inside
site-packages, which is NOT on Windows' DLL search path — so ctranslate2 builds
a CUDA model happily and then dies on the first encode with:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Call `enable()` BEFORE importing faster_whisper. No-op off Windows, and safe to
call when the wheels aren't installed (STT then falls back to CPU as usual).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def enable() -> list[str]:
    """Register every nvidia/*/bin directory in this venv. Returns those added."""
    if sys.platform != "win32":
        return []

    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return []

    added: list[str] = []
    for bin_dir in sorted(nvidia.glob("*/bin")):
        if not any(bin_dir.glob("*.dll")):
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            continue
        # PATH too: ctranslate2 resolves some libraries through it.
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(str(bin_dir))
    return added
