from __future__ import annotations

import subprocess
import sys
import platform


TORCH_INDEX_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
}

DEFAULT_TORCH_SPEC = "torch>=2.2,<2.12"


def torch_install_command(
    variant: str = "cpu",
    *,
    python_executable: str | None = None,
    torch_spec: str = DEFAULT_TORCH_SPEC,
    upgrade: bool = True,
) -> list[str]:
    if variant not in TORCH_INDEX_URLS:
        valid = ", ".join(sorted(TORCH_INDEX_URLS))
        raise ValueError(f"Unknown torch variant {variant!r}; expected one of: {valid}")
    if sys.platform == "darwin" and variant != "cpu":
        raise ValueError("CUDA wheels are not available on macOS; use --variant cpu")

    python_executable = python_executable or sys.executable
    command = [python_executable, "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    command.append(torch_spec)
    # The last official Intel-Mac wheels are PyTorch 2.2.x (NumPy 1.x ABI).
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        command.extend(["numpy<2", "mdtraj<1.11"])
    # macOS wheels are distributed through PyPI, not the CPU-only wheel index.
    index = "https://pypi.org/simple" if sys.platform == "darwin" else TORCH_INDEX_URLS[variant]
    command.extend(["--index-url", index])
    return command


def install_torch(
    variant: str = "cpu",
    *,
    python_executable: str | None = None,
    torch_spec: str = DEFAULT_TORCH_SPEC,
    upgrade: bool = True,
    dry_run: bool = False,
) -> list[str]:
    command = torch_install_command(
        variant,
        python_executable=python_executable,
        torch_spec=torch_spec,
        upgrade=upgrade,
    )
    if dry_run:
        return command

    subprocess.run(command, check=True)
    return command
