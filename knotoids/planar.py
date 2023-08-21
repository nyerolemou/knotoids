from typing import List, NamedTuple, Tuple

import numpy as np
from grapher import SphericalNode


class PlanarNode(NamedTuple):
    index: int
    position: np.array


class PlanarGraph:
    def __init__(self, spherical_nodes: List[SphericalNode], edges: List[Tuple]):
        self.nodes = self._project(spherical_nodes)
        self.edges = edges

    def _project(self, spherical_nodes: List[SphericalNode]) -> List[PlanarNode]:
        """
        Projects the spherical nodes onto the plane.
        """
        pass
