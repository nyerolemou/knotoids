import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from . import distribution
from .config import Config


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
    output: Annotated[
        Path, typer.Option(..., "--output", "-o", help="Path to output file.")
    ],
    path_to_knoto_id: Annotated[
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
):
    """
    Compute the knotoid distribution of a piecewise linear curve.
    """
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    if not Path.joinpath(path_to_knoto_id, "bin/polynomial_invariant").exists():
        logging.error(
            f"Problem with installation of Knoto-ID. {Path.joinpath(path_to_knoto_id, 'bin/polynomial_invariant')} not found."
        )
        raise RuntimeError(path_to_knoto_id)
    config = Config(source, output, path_to_knoto_id)
    distribution.compute_distribution(config=config)


if __name__ == "__main__":
    typer.run(main)
