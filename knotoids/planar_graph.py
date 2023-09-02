import collections
import functools
import math
from typing import Dict, Generator, List

import numpy as np
import shapely

from .graph import Edge, Face, PlanarNode, PlanarNodeDict, SphericalNodeDict


class PlanarGraph:
    def __init__(
        self,
        nodes: PlanarNodeDict,
        edges: List[Edge],
    ):
        self.nodes = nodes
        self.edges = edges
        self.clockwise_adjacency_dict = self._clockwise_adjacency_dict

    @classmethod
    def from_spherical_graph(
        cls, spherical_nodes: SphericalNodeDict, edges: List[Edge]
    ):
        """
        Constructs a planar graph from a spherical graph.
        """
        planar_nodes = {
            key: PlanarNode(
                index=node.index, position=stereographic_project(node.position)
            )
            for key, node in spherical_nodes.items()
        }
        return cls(planar_nodes, edges)

    def generate_faces(self) -> Generator[Face, None, None]:
        """
        Generates the faces of the planar graph.

        # TODO: this assumes graph is connected for now
        """
        face_boundaries = self._find_boundaries()
        for boundary in face_boundaries:
            if self._exterior_edge in {
                (
                    boundary[i].index,
                    boundary[(i + 1) % len(boundary)].index,
                )
                for i in range(len(boundary))
            }:
                yield Face(boundary_nodes=boundary, is_external=True)
            else:
                internal_point = self._get_internal_point(boundary)
                yield Face(
                    boundary_nodes=boundary,
                    internal_point=internal_point,
                )

    def _find_boundaries(self) -> Generator[List[PlanarNode], None, None]:
        """
        Find the face boundaries of the planar graph.

        Algorithm:

        Until all edges have been visited exactly once in each direction:
            1. Pop an oriented edge from the set of edges \cup mirror_edges.
            2. Add the two nodes to the face.
            3. Until we reach the first node again:
                1. Find the next clockwise neighbour of the current node. This forces you to traverse the
                boundary of the face in a clockwise direction, i.e. the region bound by the sequence of
                nodes is always to the right of the oriented edges.
                2. Add the node to the face.
                3. Remove the edge from the set of edges \cup mirror_edges.

        The algorithm is guaranteed to terminate because the graph is finite, and in a leafless graph each
        edge will be traversed exactly once with each orientation.
        """
        mirror_edges = [(e[1], e[0]) for e in self.edges]
        edges_and_mirrors = set(self.edges + mirror_edges)
        # stopping criterion for face-finding algorithm is that all edges have been visited
        # exactly once in each direction
        while len(edges_and_mirrors) > 0:
            face_nodes = []
            first_edge = edges_and_mirrors.pop()
            v, w = first_edge
            face_nodes.append(self.nodes[v])
            face_nodes.append(self.nodes[w])
            while w != first_edge[0]:
                next_node = self._next_clockwise_neighbour(w, v)
                edges_and_mirrors.remove((w, next_node))
                face_nodes.append(self.nodes[next_node])
                v, w = w, next_node
            face_nodes.pop()
            yield face_nodes

    def _next_clockwise_neighbour(
        self,
        root_node: int,
        current_neighbour: int,
    ) -> int:
        """
        Returns the next clockwise neighbour of root_node after current_neighbour.
        """
        neighbours = self.clockwise_adjacency_dict[root_node]
        return neighbours[(neighbours.index(current_neighbour) + 1) % len(neighbours)]

    @staticmethod
    def _get_internal_point(face_nodes: List[PlanarNode]) -> np.ndarray:
        """
        Returns a point in the interior of the face.
        """
        coordinates = [node.position for node in face_nodes] + [face_nodes[0].position]
        polygon = shapely.geometry.Polygon(coordinates)
        initial_guess = np.array(polygon.representative_point().coords[0])
        # TODO: is this guaranteed to be in the spherical region in the concave case?
        # Need to use embedded great circles or some triangulation of the sphere?
        return initial_guess

    def _clockwise_angle(self, planar_position: np.ndarray) -> float:
        """
        Returns the positive clockwise angle between w and positive x-axis.
        """
        angle = math.atan2(planar_position[1], planar_position[0])
        return 2 * np.pi + angle if (angle < 0) else angle

    def _clockwise_order(self, vertex: int, neighbours: List[int]) -> List[int]:
        """
        Returns list of neighbours of vertex in clockwise order.
        """
        neighbour_array = np.asarray([self.nodes[n].position for n in neighbours])
        vertex_position = self.nodes[vertex].position
        vectors = (neighbour_array - vertex_position[None, :]) / np.linalg.norm(
            (neighbour_array - vertex_position[None, :]), axis=0
        )
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles[angles < 0] = 2 * np.pi + angles[angles < 0]
        sorted_neighbours = [neighbours[i] for i in np.argsort(-angles)]
        return sorted_neighbours

    @functools.cached_property
    def _clockwise_adjacency_dict(self) -> Dict[int, List[int]]:
        """
        Returns a dictionary of node indices and their neighbours in clockwise order.
        """
        adjacency_dict = collections.defaultdict(list)
        for source, target in self.edges:
            adjacency_dict[source].append(target)
            adjacency_dict[target].append(source)
        for node in adjacency_dict:
            # if node has degree 2, it is already in clockwise order
            if len(adjacency_dict[node]) == 2:
                continue
            adjacency_dict[node] = self._clockwise_order(node, adjacency_dict[node])
        return adjacency_dict

    @functools.cached_property
    def _exterior_edge(self) -> Edge:
        """
        Returns an edge in the exterior face.

        # TODO: Assumes the graph is connected, and edges are embedded as straight lines.

        Algorithm:
        1. Find the node v with the smallest x-coordinate.
        2. Find v's steepest positive neighbour w.

        The oriented edge (v,w) is guaranteed to be in the exterior face.
        """
        v = min(self.nodes.values(), key=lambda node: node.position[0])
        neighbour_positions = np.asarray(
            [self.nodes[n].position for n in self.clockwise_adjacency_dict[v.index]]
        )
        gradients = (neighbour_positions[:, 1] - v.position[1]) / (
            neighbour_positions[:, 0] - v.position[0]
        )
        steepest_neighbour = self.clockwise_adjacency_dict[v.index][
            np.argmax(gradients)
        ]
        return (v.index, steepest_neighbour)


def stereographic_project(spherical_position: np.ndarray) -> np.ndarray:
    """
    Projects the spherical nodes onto the plane.

    Use stereographic projection to map the nodes from the sphere to the plane.

    Parameters
    ----------
    spherical_position : np.ndarray
        The spherical coordinates of a single node.
    """
    return np.array(
        [
            spherical_position[0] / (1 - spherical_position[2]),
            spherical_position[1] / (1 - spherical_position[2]),
        ]
    )
