from pathlib import Path
from typing import NamedTuple


class Config(NamedTuple):
    """
    Configuration for each run.
    """

    source: Path
    output: Path
    path_to_knoto_id: Path
