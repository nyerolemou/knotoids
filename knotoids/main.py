import logging
from pathlib import Path

import distribution
import typer
from structures import Config
from typing_extensions import Annotated


def main(
    source: Annotated[
        str,
        typer.Option(..., "--source", "-s", help="Path to the PL curve coordinates."),
    ],
    output: Annotated[
        str, typer.Option(..., "--output", "-o", help="Path to output file.")
    ],
    path_to_knoto_id: Annotated[
        str,
        typer.Option(..., "--knoto-id", "-k", help="Path to Knoto-ID root directory."),
    ],
):
    """
    Compute the knotoid distribution of a piecewise linear curve.
    """
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    source = Path(source)
    output = Path(output)
    path_to_knoto_id = Path(path_to_knoto_id)
    if not source.exists():
        logging.error(f"{source} does not exist. Check source path.")
        raise FileNotFoundError
    if source.suffix not in set(".txt", ".npy"):
        logging.error(
            f"Unsupported file extension: {source.suffix}.\nSource must be a .npy or .txt file."
        )
        raise ValueError
    if not path_to_knoto_id.exists():
        logging.error(f"Knoto-ID dir not found in {path_to_knoto_id}.")
        raise FileNotFoundError
    if not Path.joinpath(path_to_knoto_id, "bin/polynomial_invariant").exists():
        logging.error(
            f"Problem with installation of Knoto-ID. {Path.joinpath(path_to_knoto_id, 'bin/polynomial_invariant')} not found."
        )
        raise FileNotFoundError
    config = Config(source, output, path_to_knoto_id)
    distribution.compute_distribution(config=config)


if __name__ == "__main__":
    typer.run(main)
