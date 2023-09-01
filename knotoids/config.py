from pathlib import Path
from typing import NamedTuple, Optional


class Config(NamedTuple):
    """
    Configuration for each run.

    source: path to picewise-linear curve coordinates
    output: path to output file
    knoto_id_root: path to Knoto-ID root directory
    verbose: indicates whether to include region coordinates in output
    """

    source: Path
    knoto_id_root: Path
    verbose: bool
    output: Optional[Path] = None
