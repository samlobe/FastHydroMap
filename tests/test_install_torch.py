import sys

from FastHydroMap.cli import _build_parser, main
from FastHydroMap.install_torch import torch_install_command


def test_torch_install_command_cpu_uses_cpu_index():
    command = torch_install_command("cpu", python_executable="/tmp/python")
    assert command[:4] == ["/tmp/python", "-m", "pip", "install"]
    assert "--upgrade" in command
    assert "torch>=2.2,<2.8" in command
    assert command[-1] == "https://download.pytorch.org/whl/cpu"


def test_cli_install_torch_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fasthydromap", "install-torch", "--variant", "cpu", "--dry-run"],
    )

    main()

    out = capsys.readouterr().out
    assert "Torch install command:" in out
    assert "https://download.pytorch.org/whl/cpu" in out


def test_install_torch_subparser_is_marked_as_advanced():
    parser = _build_parser()
    subparsers_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    install_torch_parser = subparsers_action.choices["install-torch"]
    assert "advanced/manual helper" in install_torch_parser.description
