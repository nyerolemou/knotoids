import logging
from collections import defaultdict
from typing import List

import numpy as np
import planar
import region
from grapher import Grapher
from structures import Config, Region


def compute_distribution(config: Config) -> None:
    file_extension = config.suffix
    if file_extension == ".txt":
        pl_curve = np.loadtxt(config.source)
    else:
        pl_curve = np.load(config.source)

    graph = Grapher(pl_curve).compute_graph()
    planar_graph = planar.PlanarGraph(*graph)
    regions = region.generate_regions_from_faces(
        planar_graph.generate_faces(), config=config
    )
    # TODO: finish implementing. distribution summary, plots, etc.


def _summarise_distribution(regions: List[Region]) -> None:
    """
    Summarise the knotoid distribution of the curve.
    """
    distribution = defaultdict(float)
    for region in regions:
        distribution[region.knotoid_class] += region.area
    sorted_distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    s = ""
    for knotoid_class, proportion in sorted_distribution:
        s += f"{knotoid_class.value}: {round(proportion, 4)}\n"
    s += f"Total: {round(sum(distribution.values()), 4)}"
    logging.info(f"Knotoid distribution:\n{s}")


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
