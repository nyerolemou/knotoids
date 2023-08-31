import grapher
import typer
from typing_extensions import Annotated


def main(
    source: Annotated[
        str,
        typer.Option(..., "--source", "-s", help="Path to the PL curve coordinates."),
    ],
    target: Annotated[
        str, typer.Option(..., "--target", "-t", help="Path to output file.")
    ],
    path_to_ki: Annotated[
        str,
        typer.Option(..., "--knoto-id", "-k", help="Path to Knoto-ID root directory."),
    ],
):
    """
    Compute the knotoid distribution of your piecewise linear curve.
    """
    # TODO: check curve file exists
    # TODO: check Knoto-ID root directory exists
    # grapher = grapher.Grapher(source, target, path_to_ki)
    pass


if __name__ == "__main__":
    typer.run(main)
