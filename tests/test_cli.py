import pytest
from click.testing import CliRunner

from tmqi.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def off_pair(data_dir):
    return str(data_dir / "off.png"), str(data_dir / "off_ldr.png")


def test_default_invocation(runner, off_pair):
    result = runner.invoke(main, list(off_pair))
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("Q: ")
    assert len(result.stdout.strip().splitlines()) == 1


@pytest.mark.parametrize(
    ("flag", "label"),
    [("-S", "S: "), ("-N", "N: "), ("-L", "S_locals: ")],
)
def test_report_flags_add_fields(runner, off_pair, flag, label):
    result = runner.invoke(main, [*off_pair, flag])
    assert result.exit_code == 0
    assert label in result.stdout


def test_no_report_q_omits_q(runner, off_pair):
    result = runner.invoke(main, [*off_pair, "-q"])
    assert result.exit_code == 0
    assert "Q: " not in result.stdout


def test_combined_report_flags_preserve_order(runner, off_pair):
    result = runner.invoke(main, [*off_pair, "-Q", "-S", "-L", "-N"])
    assert result.exit_code == 0
    line = result.stdout.strip()
    assert line.index("Q: ") < line.index("S: ") < line.index("N: ") < line.index("S_locals: ")


def test_quiet_omits_labels(runner, off_pair):
    result = runner.invoke(main, [*off_pair, "-Q", "-S", "-N", "--quiet"])
    assert result.exit_code == 0
    assert "Q:" not in result.stdout
    assert len(result.stdout.strip().split()) == 3


def test_verbose_is_default(runner, off_pair):
    result = runner.invoke(main, [*off_pair, "--verbose"])
    assert result.exit_code == 0
    assert "Q: " in result.stdout


@pytest.mark.parametrize("precision", [2, 8])
def test_precision_controls_decimals(runner, off_pair, precision):
    result = runner.invoke(main, [*off_pair, "-p", str(precision)])
    assert result.exit_code == 0
    value = result.stdout.strip().removeprefix("Q: ")
    decimals = len(value.split(".")[-1]) if "." in value else 0
    assert decimals <= precision


def test_revised_changes_output(runner, off_pair):
    default_result = runner.invoke(main, list(off_pair))
    revised_result = runner.invoke(main, [*off_pair, "--revised"])
    assert default_result.exit_code == revised_result.exit_code == 0
    assert default_result.stdout != revised_result.stdout


@pytest.mark.parametrize("maptype", ["float32", "png"])
def test_report_maps_writes_five_files_with_prefix(
    runner, off_pair, tmp_path, monkeypatch, maptype
):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, [*off_pair, "-M", "-t", maptype, "-m", "prefix_", "-q"])
    assert result.exit_code == 0, result.stderr
    written = sorted(tmp_path.glob(f"prefix_*.{maptype}"))
    assert len(written) == 5


def test_raw_input_matches_png_input_closely(runner, data_dir):
    # The gray raw/PNG fixtures aren't numerically identical (see
    # test_image_io.py::test_raw_gray_matches_png_gray_mostly and FINDINGS.md), so this
    # compares Q with a tolerance rather than requiring exact stdout equality.
    png_result = runner.invoke(
        main, [str(data_dir / "test.png"), str(data_dir / "test_ldr.png"), "-g", "-p", "6"]
    )
    raw_result = runner.invoke(
        main,
        [
            str(data_dir / "test.float32"),
            str(data_dir / "test_ldr.float32"),
            "-i",
            "float32",
            "-W",
            "396",
            "-H",
            "561",
            "-g",
            "-p",
            "6",
        ],
    )
    assert png_result.exit_code == raw_result.exit_code == 0
    png_q = float(png_result.stdout.removeprefix("Q: ").strip())
    raw_q = float(raw_result.stdout.removeprefix("Q: ").strip())
    assert png_q == pytest.approx(raw_q, abs=1e-3)


def test_missing_argument_exits_2_stdout_empty(runner, data_dir):
    result = runner.invoke(main, [str(data_dir / "test.png")])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Missing argument" in result.stderr


def test_nonexistent_file_exits_2(runner, data_dir, tmp_path):
    missing = tmp_path / "nope.png"
    result = runner.invoke(main, [str(missing), str(data_dir / "test_ldr.png")])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_input_type_without_width_height_is_usage_error(runner, data_dir):
    result = runner.invoke(
        main,
        [str(data_dir / "test.float32"), str(data_dir / "test_ldr.float32"), "-i", "float32"],
    )
    assert result.exit_code == 2
    assert "width" in result.stderr and "height" in result.stderr


def test_logfile_writes_debug_lines_stdout_still_one_line(runner, off_pair, tmp_path):
    logfile = tmp_path / "run.log"
    result = runner.invoke(main, [*off_pair, "--logfile", str(logfile), "--loglevel", "DEBUG"])
    assert result.exit_code == 0
    assert len(result.stdout.strip().splitlines()) == 1
    assert logfile.exists()
    assert "DEBUG" in logfile.read_text()


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_help(runner, help_flag):
    result = runner.invoke(main, [help_flag])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


@pytest.mark.slow
def test_raw_rgb_matches_png_rgb_via_cli(runner, data_dir):
    png_result = runner.invoke(main, [str(data_dir / "test.png"), str(data_dir / "test_ldr.png")])
    raw_result = runner.invoke(
        main,
        [
            str(data_dir / "rgb_test.float32"),
            str(data_dir / "rgb_test_ldr.float32"),
            "-i",
            "float32",
            "-W",
            "396",
            "-H",
            "561",
        ],
    )
    assert png_result.exit_code == raw_result.exit_code == 0
    assert png_result.stdout == raw_result.stdout
