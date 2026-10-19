# Publishing FastHydroMap

See [the release checklist](RELEASING.md) for approval gates, versioning,
PyPI and ChimeraX Toolshed publication, and preserving old releases.

```bash
python -m pytest -q
./scripts/build_release.sh
./scripts/test_wheel_install.sh
```

Each helper uses a fresh temporary directory and prints its location. Build
artifacts and verification outputs are retained for review; no existing test
environment is recursively deleted. The helpers do not upload packages.
