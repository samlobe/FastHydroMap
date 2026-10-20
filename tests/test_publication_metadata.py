"""Distribution metadata retains upstream notices and usable PyPI links."""

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_data_license_is_declared_and_available():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT AND LicenseRef-VIPARR-FF"
    assert set(project["license-files"]) == {
        "LICENSE", "licenses/VIPARR-FF-LICENSE.txt",
    }
    for name in project["license-files"]:
        assert (ROOT / name).is_file()
    notice = (ROOT / "licenses/VIPARR-FF-LICENSE.txt").read_text()
    assert "Copyright 2018, D. E. Shaw Research. All rights reserved." in notice
    assert "Redistributions in binary form must reproduce" in notice


def test_pypi_readme_links_do_not_depend_on_a_checkout():
    readme = (ROOT / "README.md").read_text()
    assert "python -m pip install fasthydromap" in readme
    assert "python -m pip install -e ." not in readme
    assert "not yet available on PyPI" not in readme
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    assert links
    assert all(link.startswith("https://") for link in links)
