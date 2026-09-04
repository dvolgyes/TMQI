"""Command-line interface for computing the TMQI/TMQIr metrics."""

import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

from TMQI.image_io import RAW_DTYPES, img_read, write_map
from TMQI.metric import TMQI, TMQIr, TMQIResult

_LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


def _configure_logging(loglevel: str, logfile: Path | None) -> None:
    logger.enable("TMQI")
    logger.remove()
    logger.add(sys.stderr, level=loglevel)
    if logfile is not None:
        logger.add(logfile, level=loglevel)


def format_report(
    result: TMQIResult,
    *,
    precision: int,
    quiet: bool,
    report_q: bool,
    report_s: bool,
    report_n: bool,
    report_sl: bool,
) -> str:
    """Render the requested descriptors as the single-line report the CLI prints."""
    q, s, n = (np.round(x, precision) for x in (result.Q, result.S, result.N))
    s_local_str = " ".join(str(x) for x in np.round(result.s_local, precision))

    parts = []
    if report_q:
        parts.append(("" if quiet else "Q: ") + str(q))
    if report_s:
        parts.append(("" if quiet else "S: ") + str(s))
    if report_n:
        parts.append(("" if quiet else "N: ") + str(n))
    if report_sl:
        parts.append(("" if quiet else "S_locals: ") + s_local_str)
    return " ".join(parts)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hdr_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ldr_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-t", "--type", "maptype", default="float32", show_default=True, help="s_map file type"
)
@click.option(
    "-m",
    "--smap_file",
    "--smap-file",
    "smap",
    default="s_map_",
    show_default=True,
    help="s_map file name prefix",
)
@click.option(
    "-p",
    "--precision",
    type=int,
    default=4,
    show_default=True,
    help="precision (number of decimals)",
)
@click.option("-W", "--width", type=int, default=None, help="image width (mandatory for RAW files)")
@click.option(
    "-H", "--height", type=int, default=None, help="image height (mandatory for RAW files)"
)
@click.option(
    "-i",
    "--input_type",
    "--input-type",
    "input_type",
    type=click.Choice(sorted(RAW_DTYPES)),
    default=None,
    help="dtype of raw input images (default: regular image files)",
)
@click.option("-g", "--gray", is_flag=True, default=False, help="gray input (lightness/brightness)")
@click.option(
    "-Q/-q", "--report-Q/--no-report-Q", "report_q", default=True, help="report quality index"
)
@click.option(
    "-S/-s",
    "--report-S/--no-report-S",
    "report_s",
    default=False,
    help="report structural similarity",
)
@click.option(
    "-L/-l", "--report-SL/--no-report-SL", "report_sl", default=False, help="report maps (S_locals)"
)
@click.option(
    "-N/-n", "--report-N/--no-report-N", "report_n", default=False, help="report naturalness"
)
@click.option(
    "-M",
    "--report-MAPS",
    "--report-maps",
    "report_maps",
    is_flag=True,
    default=False,
    help="report maps",
)
@click.option(
    "--quiet/--verbose", "quiet", default=False, help="suppress variable names in the report"
)
@click.option("-r", "--revised", is_flag=True, default=False, help="Enable revised TMQI")
@click.option(
    "--logfile",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="optional log file",
)
@click.option(
    "--loglevel",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
    help="logging verbosity",
)
@click.version_option(package_name="tmqi-revised")
def main(
    hdr_image: Path,
    ldr_image: Path,
    maptype: str,
    smap: str,
    precision: int,
    width: int | None,
    height: int | None,
    input_type: str | None,
    gray: bool,
    report_q: bool,
    report_s: bool,
    report_sl: bool,
    report_n: bool,
    report_maps: bool,
    quiet: bool,
    revised: bool,
    logfile: Path | None,
    loglevel: str,
) -> None:
    """Compute the Tone Mapped Image Quality Index for HDR_IMAGE against LDR_IMAGE."""
    _configure_logging(loglevel, logfile)

    if input_type is not None and (width is None or height is None):
        raise click.UsageError("-W/--width and -H/--height are required with -i/--input_type")

    shape = (width, height) if input_type is not None else None
    hdr = img_read(hdr_image, gray=gray, shape=shape, dtype=input_type)
    ldr = img_read(ldr_image, gray=gray, shape=shape, dtype=input_type)
    logger.debug("read {} shape={}", hdr_image, hdr.shape)
    logger.debug("read {} shape={}", ldr_image, ldr.shape)

    metric = TMQIr() if revised else TMQI()
    logger.debug("using {} metric", metric.name)
    result = metric(hdr, ldr)

    click.echo(
        format_report(
            result,
            precision=precision,
            quiet=quiet,
            report_q=report_q,
            report_s=report_s,
            report_n=report_n,
            report_sl=report_sl,
        )
    )

    if report_maps:
        for idx, s_map in enumerate(result.s_maps, start=1):
            map_path = Path(f"{smap}{idx}.{maptype}")
            write_map(map_path, s_map, maptype)
            logger.info("wrote {}", map_path)
