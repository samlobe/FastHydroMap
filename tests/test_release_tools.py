"""Non-network checks for release helper input and overwrite guards."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or os.name == "nt", reason="POSIX release helpers")


def run_helper(script, tmp_path, *arguments):
    env = {**os.environ, "PYTHON": sys.executable, "TMPDIR": str(tmp_path)}
    return subprocess.run(
        [BASH, str(ROOT / "scripts" / script), *map(str, arguments)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("script", ["build_release.sh", "test_wheel_install.sh"])
def test_release_helper_rejects_extra_arguments(tmp_path, script):
    result = run_helper(script, tmp_path, "one", "two")
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not list(tmp_path.iterdir())


def test_build_preserves_existing_artifacts(tmp_path):
    destination = tmp_path / "dist"
    destination.mkdir()
    existing = destination / "previous.whl"
    existing.write_bytes(b"retain this artifact")
    result = run_helper("build_release.sh", tmp_path, destination)
    assert result.returncode != 0
    assert "Output directory must be empty" in result.stderr
    assert existing.read_bytes() == b"retain this artifact"


def test_wheel_helper_rejects_non_wheel_before_install(tmp_path):
    invalid = tmp_path / "notes.txt"
    invalid.write_text("not a distribution")
    result = run_helper("test_wheel_install.sh", tmp_path, invalid)
    assert result.returncode != 0
    assert "Expected a wheel file" in result.stderr
    assert not list(tmp_path.glob("fasthydromap-wheel.*/env"))
