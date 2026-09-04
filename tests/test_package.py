import subprocess
import sys
from importlib.metadata import version


def test_documented_import_contract():
    from TMQI import TMQI, TMQIr  # noqa: F401


def test_version_matches_installed_distribution():
    import TMQI

    assert TMQI.__version__ == version("tmqi-revised")


def test_metadata_dunders_present():
    import TMQI

    for name in (
        "__title__",
        "__summary__",
        "__uri__",
        "__author__",
        "__email__",
        "__upstream_uri__",
        "__upstream_doi__",
        "__upstream_ref__",
    ):
        assert getattr(TMQI, name)


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "TMQI", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
