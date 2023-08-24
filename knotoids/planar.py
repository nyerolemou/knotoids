import collections
import functools
import math
from typing import Dict, Iterable, List

import numpy as np
import shapely
from structures import Edge, Face, PlanarNode, SphericalNodeDict


class PlanarGraph:
    def __init__(
        self,
        spherical_nodes: SphericalNodeDict,
        edges: List[Edge],
    ):
        self.nodes = {
            key: PlanarNode(
                index=node.index, position=PlanarGraph._project(node.position)
            )
            for key, node in spherical_nodes.items()
        }
        self.edges = edges
        self.clockwise_adjacency_dict = self._get_clockwise_adjacency_dict

    def generate_faces(self) -> Iterable[Face]:
        """
        Find the faces of the planar graph.

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
        # if graph is disconnected, the external face of the plane is the union of exactly two exernal faces
        external_boundaries = []
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
            if PlanarGraph._is_external_face(face_nodes):
                external_boundaries.append(face_nodes)
            else:
                point_in_region = self._get_internal_point(face_nodes)
                yield Face(point_in_region, face_nodes)
        if len(external_boundaries) == 1:
            point_in_region = self._get_internal_point(
                external_boundaries[0], is_external=True
            )
            yield Face(point_in_region, external_boundaries[0])
        else:
            # when graph is disconnected: the external face of the plane is the
            # union of the external faces of each connected component
            first_boundary, second_boundary = (
                external_boundaries[0],
                external_boundaries[1],
            )
            point_in_region = self._get_internal_point(
                first_boundary, second_boundary, is_external=True
            )
            yield Face(
                point_in_region,
                first_boundary,
                second_boundary_nodes=second_boundary,
            )

    # TODO: refactor to do in constant time.
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

    def _get_internal_point(
        self,
        face_nodes: List[PlanarNode],
        other_boundary_nodes: List[PlanarNode] = None,
        is_external: bool = False,
    ) -> np.ndarray:
        """
        Returns a point in the interior of the face.
        """
        if not is_external:
            coordinates = [node.position for node in face_nodes] + [
                face_nodes[0].position
            ]
            polygon = shapely.geometry.Polygon(coordinates)
            initial_guess = np.array(polygon.representative_point().coords[0])
            # TODO: is this guaranteed to be in the spherical region in the concave case?
            return initial_guess
        else:
            # 1. Find the point p in the external boundar(ies) furthest from the origin
            # 2. Move away from this extremal point in direction p
            if not other_boundary_nodes:
                all_coordinates = np.asarray([node.position for node in face_nodes])
            else:
                all_coordinates = np.asarray(
                    [node.position for node in face_nodes + other_boundary_nodes]
                )
            # find point in all_coordinates furthest from origin
            distances = np.linalg.norm(all_coordinates, axis=1)
            extremal_point = all_coordinates[np.argmax(distances)]
            # move away from extremal point in direction of extremal point
            # TODO: is this far enough?
            return extremal_point + extremal_point / np.linalg.norm(extremal_point)

    # TODO: merge with _clockwise_order?
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
    def _get_clockwise_adjacency_dict(self) -> Dict[int, List[int]]:
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

    @staticmethod
    def _is_external_face(vertices: List[PlanarNode]) -> bool:
        # TODO: fix this as it doesn't seem to be working properly...
        n = len(vertices)
        for i in range(n):
            if PlanarGraph._is_counterclockwise(
                vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]
            ):
                return False
        return True

    @staticmethod
    def _is_counterclockwise(a: PlanarNode, b: PlanarNode, c: PlanarNode) -> bool:
        """
        Returns True if the oriented edge (a,b) is counterclockwise with respect to c.
        """
        x = (c.position[1] - a.position[1]) * (b.position[0] - a.position[0]) > (
            b.position[1] - a.position[1]
        ) * (c.position[0] - a.position[0])
        return x

    @staticmethod
    def _project(spherical_position: np.ndarray) -> np.ndarray:
        """
        Projects the spherical nodes onto the plane.

        Use stereographic projection to map the nodes from the sphere to the plane.
        """
        return np.array(
            [
                spherical_position[0] / (1 - spherical_position[2]),
                spherical_position[1] / (1 - spherical_position[2]),
            ]
        )


if __name__ == "__main__":
    # development code
    import plotting
    import structures

    spherical_nodes = {
        0: structures.SphericalNode(index=0, position=np.array([0, -1, 0])),
        1: structures.SphericalNode(index=1, position=np.array([-1, 0, 0])),
        2: structures.SphericalNode(index=2, position=np.array([0, 1, 0])),
        3: structures.SphericalNode(
            index=3, position=np.array([0.2, 0.3, np.sqrt(1 - 0.2**2 - 0.3**2)])
        ),
        4: structures.SphericalNode(
            index=4, position=np.array([0.2, 0.4, np.sqrt(1 - 0.2**2 - 0.4**2)])
        ),
        5: structures.SphericalNode(
            index=5, position=np.array([0.3, -0.2, np.sqrt(1 - 0.3**2 - 0.2**2)])
        ),
    }
    connected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5), (0, 5)]
    # disconnected_edges = [(1, 2), (2, 3), (1, 3), (4, 0), (4, 5), (0, 5)]
    connected_graph = PlanarGraph(
        spherical_nodes=spherical_nodes, edges=connected_edges
    )
    plotting.plot_planar_from_nodes_and_edges(
        connected_graph.nodes, connected_graph.edges
    )
    for face in connected_graph.generate_faces():
        print([node.index for node in face.boundary_nodes])
        print(face.internal_point)
