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
):
    """
    Compute the knotoid distribution of your piecewise linear curve.
    """
    print(f"Source: {source}\nTarget: {target}")


if __name__ == "__main__":
    typer.run(main)
