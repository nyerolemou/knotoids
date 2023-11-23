from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .knotoid_class import KnotoidClass


class Node(NamedTuple):
    index: int
    position: np.ndarray


SphericalNode = Node
PlanarNode = Node
Edge = Tuple[int, int]
SphericalNodeDict = Dict[int, SphericalNode]
PlanarNodeDict = Dict[int, PlanarNode]

BaseRegion = namedtuple(
    "BaseRegion",
    ["internal_point", "boundary_nodes", "knotoid_class", "area", "is_external"],
)


class Region(BaseRegion):
    """
    A region of the sphere bounded by a simple closed curve.

    The order of the nodes is important; it determines the edges of the boundary.

    The nodes do not form a closed curve; v_0 != v_n, and the edge v_n->v_0 exists.
    """

    internal_point: np.ndarray
    boundary_nodes: List[SphericalNode]
    knotoid_class: KnotoidClass
    area: float
    # second_boundary_nodes: Optional[List[SphericalNode]] = None
    is_external: bool = False

    def __new__(
        cls, internal_point, boundary_nodes, knotoid_class, area, is_external=False
    ):
        # Check if internal_point has unit norm
        if not np.isclose(np.linalg.norm(internal_point), 1.0):
            raise ValueError("internal_point must have a unit norm")

        return super(Region, cls).__new__(
            cls, internal_point, boundary_nodes, knotoid_class, area, is_external
        )


class Face(NamedTuple):
    """
    A region of the plane bounded by a simple closed curve.

    The order of the nodes is important. It determines the orientation of the region.

    The nodes do not form a closed curve; v_0 != v_n, and the edge v_n->v_0 exists.
    """

    boundary_nodes: List[PlanarNode]
    internal_point: Optional[np.ndarray] = None
    # second_boundary_nodes: Optional[List[PlanarNode]] = None
    is_external: bool = False
