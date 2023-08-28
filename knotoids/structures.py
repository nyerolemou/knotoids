from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from knotoids.knotoid_class import KnotoidClass


class Node(NamedTuple):
    index: int
    position: np.ndarray


SphericalNode = Node
PlanarNode = Node
Edge = Tuple[int, int]
SphericalNodeDict = Dict[int, SphericalNode]
PlanarNodeDict = Dict[int, PlanarNode]


class Region(NamedTuple):
    """
    A region of the sphere bounded by a simple closed curve.

    The order of the nodes is important; it determines the orientation of the region.
    """

    internal_point: np.ndarray
    boundary_nodes: List[SphericalNode]
    knotoid_class: KnotoidClass
    area: float
    second_boundary_nodes: Optional[List[SphericalNode]] = None


class Face(NamedTuple):
    """
    A region of the plane bounded by a simple closed curve.

    The order of the nodes is important. It determines the orientation of the region.
    """

    boundary_nodes: List[PlanarNode]
    internal_point: Optional[np.ndarray] = None
    # second_boundary_nodes: Optional[List[PlanarNode]] = None
    is_external: bool = False
