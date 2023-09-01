import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import planar_graph, region
from .config import Config
from .graph import Region
from .spherical_graph import SphericalGraph


def compute_distribution(config: Config) -> None:
    file_extension = config.source.suffix
    if file_extension in [".txt", ".xyz"]:
        pl_curve = np.loadtxt(config.source)
    elif file_extension == ".npy":
        pl_curve = np.load(config.source)
    else:
        raise ValueError(f"Unsupported file extension {file_extension}")

    graph = SphericalGraph(pl_curve).compute_graph()
    embedded_graph = planar_graph.PlanarGraph(*graph)
    regions = region.generate_regions_from_faces(
        embedded_graph.generate_faces(), config=config
    )
    _summarise_distribution(list(regions), verbose=config.verbose, output=config.output)
    # TODO: add plot generation


def _summarise_distribution(
    regions: List[Region], verbose: bool, output: Optional[Path]
) -> None:
    """
    Summarise the knotoid distribution of the curve.
    """
    json_output = {"distribution": None, "regions": None}
    distribution_summary = defaultdict(float)
    region_summary = {}
    for idx, region in enumerate(regions):
        distribution_summary[region.knotoid_class.value] += region.area
        if verbose:
            region_summary[idx] = region._asdict()
    sorted_distribution = sorted(
        distribution_summary.items(), key=lambda x: x[1], reverse=True
    )
    s = ""
    for knotoid_class, proportion in sorted_distribution:
        s += f"{knotoid_class}: {round(proportion, 4)}\n"
    logging.info(f"\n\n***Knotoid distribution***\n{s}")

    if output is not None:
        json_output["distribution"] = sorted_distribution
        # TODO: serialization for regions. Currently, verbose output doesn't work.
        json_output["regions"] = region_summary
        output_file = output.with_suffix(".json")
        logging.info(f"Saving output to {output_file}")
        with open(output_file, "w") as f:
            json.dump(json_output, f)


def _subdivide_curve(curve: np.ndarray, factor: int) -> np.ndarray:
    """
    Subdivide piecewise linear curve.
    """
    new_curve = []
    for point, next_point in zip(curve[:-1], curve[1:]):
        new_curve.append(point)
        for i in range(1, factor):
            new_curve.append(point + i * (next_point - point) / factor)
    new_curve.append(curve[-1])
    return np.array(new_curve)
