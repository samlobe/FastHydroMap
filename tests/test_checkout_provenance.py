"""Byte-verified resources retain canonical line endings on Windows checkouts."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")


@pytest.mark.skipif(GIT is None, reason="Git is required to test checkout filters")
def test_windows_checkout_preserves_byte_verified_resources(tmp_path):
    repository = tmp_path / "repository"
    checkout = tmp_path / "checkout"
    repository.mkdir()

    def git(*arguments):
        return subprocess.run(
            [GIT, "-C", str(repository), *arguments],
            check=True, capture_output=True, text=True,
        )

    git("init", "--quiet")
    git("config", "core.autocrlf", "true")
    git("config", "core.eol", "crlf")
    files = (
        ".gitattributes",
        "tests/data/v2_chain.pdb",
        "tests/data/v2_chain_preparation.json",
        "tests/data/v2_shared_chain_expected.json",
        "src/FastHydroMap/v2/chemistry.json",
        "src/FastHydroMap/weights/v2/manifest.json",
        "src/FastHydroMap/weights/v2/shared16_seed17.pt",
        "training/shared_protocol.json",
    )
    for name in files:
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    # Confirm that the checkout really performs Windows newline conversion.
    (repository / "control.txt").write_bytes(b"newline control\n")
    git("add", ".")
    git("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")
    assert (checkout / "control.txt").read_bytes() == b"newline control\r\n"
    for name in files[1:]:
        assert (checkout / name).read_bytes() == (ROOT / name).read_bytes(), name
