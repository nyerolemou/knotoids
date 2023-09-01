import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from . import distribution
from .config import Config

cli = typer.Typer()


@cli.command()
def main(
    source: Annotated[
        Path,
        typer.Option(
            ...,
            "--source",
            "-s",
            help="Path to the PL curve coordinates.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    knoto_id_root: Annotated[
        Path,
        typer.Option(
            ...,
            "--knoto-id",
            "-k",
            help="Path to Knoto-ID root directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to output file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            ...,
            "--verbose",
            "-v",
            help="If True, save all region data.",
            show_default=True,
        ),
    ] = False,
):
    """
    Compute the knotoid distribution of a piecewise linear curve.
    """
    logging.basicConfig(
        level=logging.INFO,
    )
    if not Path.joinpath(knoto_id_root, "bin/polynomial_invariant").exists():
        logging.error(
            f"Problem with installation of Knoto-ID. {Path.joinpath(knoto_id_root, 'bin/polynomial_invariant')} not found."
        )
        raise RuntimeError(knoto_id_root)
    config = Config(
        source=source, knoto_id_root=knoto_id_root, output=output, verbose=verbose
    )
    distribution.compute_distribution(config=config)


if __name__ == "__main__":
    cli()
