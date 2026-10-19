import sys
import pytest

from FastHydroMap.cli import _build_parser, main
from FastHydroMap.install_torch import torch_install_command


@pytest.mark.parametrize("platform, index", [
    ("linux", "https://download.pytorch.org/whl/cpu"),
    ("win32", "https://download.pytorch.org/whl/cpu"),
    ("darwin", "https://pypi.org/simple"),
])
def test_torch_install_command_cpu_uses_platform_index(monkeypatch, platform, index):
    monkeypatch.setattr(sys, "platform", platform)
    command = torch_install_command("cpu", python_executable="/tmp/python")
    assert command[:4] == ["/tmp/python", "-m", "pip", "install"]
    assert "--upgrade" in command
    assert "torch>=2.2,<2.12" in command
    assert command[-1] == index


def test_cli_install_torch_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fasthydromap", "install-torch", "--variant", "cpu", "--dry-run"],
    )

    main()

    out = capsys.readouterr().out
    assert "Torch install command:" in out
    index = "https://pypi.org/simple" if sys.platform == "darwin" else "https://download.pytorch.org/whl/cpu"
    assert index in out


def test_install_torch_subparser_has_plain_description():
    parser = _build_parser()
    subparsers_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    install_torch_parser = subparsers_action.choices["install-torch"]
    assert install_torch_parser.description == "Install or replace PyTorch in the current environment."
