import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import plotting, region
from .config import Config
from .graph import Region
from .planar_graph import PlanarGraph
from .spherical_graph import SphericalGraph


def compute_distribution(config: Config) -> None:
    pl_curve = config.load_pl_curve()
    graph = SphericalGraph(pl_curve).compute_graph()
    embedded_graph = PlanarGraph.from_spherical_graph(*graph)
    regions = list(
        region.generate_regions_from_faces(
            embedded_graph.generate_faces(), config=config
        )
    )
    _summarise_distribution(regions, verbose=config.verbose, output=config.output)
    plotting.plot_spherical_regions(regions, output=config.output, curve=pl_curve)


def _summarise_distribution(
    regions: List[Region], verbose: bool, output: Optional[Path]
) -> None:
    """
    Summarise the knotoid distribution of the curve.
    """
    distribution_summary = defaultdict(float)
    region_summary = {}
    for idx, region in enumerate(regions):
        distribution_summary[region.knotoid_class.value] += region.area
        if verbose:
            region_summary[idx] = region._asdict()
    sorted_distribution = sorted(
        distribution_summary.items(), key=lambda x: x[1], reverse=True
    )
    knotoid_dist_str = "\n".join(
        f"{knotoid_class}: {round(proportion, 4)}"
        for knotoid_class, proportion in sorted_distribution
    )
    logging.info(f"\n\n***Knotoid distribution***\n{knotoid_dist_str}\n")

    if output is None:
        return

    json_output = {}
    json_output["distribution"] = sorted_distribution
    # TODO: serialization for knotoidclass - verbose output doesn't work atm.
    json_output["regions"] = region_summary
    output_file = output / "summary.json"
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
