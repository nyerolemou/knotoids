import copy
import itertools
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
import planar
import plotting
import scipy as sp
from knotoid_class import KnotoidClass
from spherical_geometry import great_circle_arc, polygon
from structures import Edge, Region, SphericalNode, SphericalNodeDict


# TODO: class is redundant? Keep graph functionality here and move region creation to separate class/mod?
class Grapher:
    """
    Computes the boundaries of the regions on the sphere, each of which corresponds to a different knotoid classification.

    Calls Knoto-ID to classify each region.
    """

    def __init__(self, pl_curve: np.ndarray):
        self.pl_curve = pl_curve

    def compute_regions(self) -> Iterable[Region]:
        """
        Returns regions, each of which corresponds to a different knotoid classification.
        """
        nodes, edges = self._compute_graph()
        planar_graph = planar.PlanarGraph(nodes, edges)
        for face in planar_graph.generate_faces():
            boundary_nodes = [
                SphericalNode(node.index, Grapher._inverse_projection(node.position))
                for node in face.boundary_nodes
            ]
            if not face.is_external:
                internal_point = Grapher._inverse_projection(face.internal_point)
            else:
                # north pole is the representative point of the external face
                internal_point = np.array([0, 0, 1])
            # TODO: area not correct for external region bounded by two curves.
            # TODO: have to add first node to end of boundary_nodes and only input positions
            # area = polygon.SphericalPolygon(boundary_nodes, internal_point).area() / (
            #     4 * np.pi
            # )
            area = 1.0
            # TODO: performance cost calling Knoto-ID for each region, especially if using subprocess
            classification = self._classify_region(internal_point)
            yield Region(internal_point, boundary_nodes, classification, area)

    def _compute_graph(self) -> Tuple[SphericalNodeDict, List[Edge]]:
        """
        Computes graph on the surface of the sphere, whose regions correspond to a (potentially different)
        knotoid classification.

        Algorithm:
        1. Find two connected graphs on the sphere formed from normalised vectors between points and each fixed endpoint.
        2. Find the intersection of these two graphs, adding new vertices and edges where required.
        3. Remove all leaves from the graph.

        Suppose self.pl_curve has n+1 vertices {v0, v1, ..., vn}.

        Then the two connected graphs, G1 and G2, each have:
        - 2n-1 nodes
        - 2n-2 edges,
        and G2 is the graph antipodal to G1.
        """
        n = self.pl_curve.shape[0] - 1
        # Get first spherical graph (second one is just antipodal)
        first_endpoint_stack = self.pl_curve[0] - self.pl_curve[1:]
        last_endpoint_stack = -(self.pl_curve[-1] - self.pl_curve[1:-1])
        first_endpoint_stack = (
            first_endpoint_stack / np.linalg.norm(first_endpoint_stack, axis=1)[:, None]
        )
        last_endpoint_stack = (
            last_endpoint_stack / np.linalg.norm(last_endpoint_stack, axis=1)[:, None]
        )
        g = np.vstack((first_endpoint_stack, last_endpoint_stack))
        # remove rows that are the same
        _, unique_idx = np.unique(np.round(g, 8), axis=0, return_index=True)
        unique_idx_sorted = np.sort(unique_idx)
        g = g[unique_idx_sorted]
        # create vertex dictionary
        nodes = {}

        # Edges and nodes per graph
        # graph1 for first endpoint and negative of last endpoint
        # graph2 for last endpoint and negative of first endpoint
        NUM_NODES = g.shape[0]  # 2 * n - 1 if no duplicates removed
        NUM_EDGES = NUM_NODES - 1

        # add vertices for graph1
        for i in range(NUM_NODES):
            nodes[i] = SphericalNode(index=i, position=g[i])
        # add vertices from antipodal curve
        IDX_OFFSET = NUM_NODES
        for i in range(NUM_NODES):
            antipodal_pos = -g[i]
            nodes[IDX_OFFSET + i] = SphericalNode(
                index=IDX_OFFSET + i, position=antipodal_pos
            )
        # edges are consecutive pairs of vertices, excluding any edges that join antipodal graphs
        edges = [(i, i + 1) for i in range(NUM_EDGES)] + [
            (i, i + 1) for i in range(IDX_OFFSET, IDX_OFFSET + NUM_EDGES)
        ]
        # find intersection of pair of antipodal curves
        nodes, edges = self._resolve_intersections(nodes, edges)
        # remove all leaves
        nodes, edges = Grapher._remove_leaves(nodes, edges)
        return nodes, edges

    def _resolve_intersections(
        self, nodes: SphericalNodeDict, edges: List[Edge]
    ) -> Tuple[SphericalNodeDict, List[Edge]]:
        """
        Finds the intersection of two graphs, adding new vertices and edges where required.

        Credit to Alex for the algorithm. O(n^2), where n=#number of edges.

        Algorithm:
        1. For each pair of edges, check for intersection of the corresponding arcs.
        2. If there is an intersection:
            - Add a new vertex at the intersection.
            - Track which new vertices are connected to which edges.
        3. For each edge that has an intersection:
            - Sort the new vertices by distance from the first vertex of the edge (orientation doesn't matter).
            - Add new edges between the new vertices, i.e. interpolate old edge with new vertices.
            - Remove the old edge.
        """
        nodes = copy.copy(nodes)
        edges = copy.copy(edges)
        intersection_tracker = defaultdict(list)
        for first_edge, second_edge in itertools.combinations(edges, 2):
            # TODO: why invalid value encountered in intersection?
            intersection = great_circle_arc.intersection(
                nodes[first_edge[0]].position,
                nodes[first_edge[1]].position,
                nodes[second_edge[0]].position,
                nodes[second_edge[1]].position,
            )
            if not np.isnan(intersection).any():
                m = len(nodes)
                nodes[m] = SphericalNode(index=m, position=intersection)
                # keep track of which new nodes are connected to which edges
                intersection_tracker[first_edge].append(m)
                intersection_tracker[second_edge].append(m)
        for edge in intersection_tracker:
            # note: we always assume an edge is the *minor* arc
            # note: intersection_tracker[edge] is a list of node indices, not SphericalNode objects
            new_nodes = intersection_tracker[edge]
            new_nodes.extend([edge[0], edge[1]])
            # sort the new nodes by distance from the first vertex of the edge
            # TODO: intersections at the end of a strand?
            new_node_distances_to_first_vertex = [
                np.arccos(
                    np.clip(
                        np.dot(nodes[edge[0]].position, nodes[new_node].position),
                        a_min=-1.0,
                        a_max=1.0,
                    )
                )
                for new_node in new_nodes
            ]
            new_nodes = [
                new_node
                for _, new_node in sorted(
                    zip(new_node_distances_to_first_vertex, new_nodes)
                )
            ]
            # add new edges and remove old edge
            edges.extend(
                [(new_nodes[i], new_nodes[i + 1]) for i in range(len(new_nodes) - 1)]
            )
            edges.remove(edge)
        return nodes, edges

    def _classify_region(self, point: np.ndarray) -> KnotoidClass:
        """
        Finds the knotoid classification of each region using Knoto-ID.
        """
        # TODO: call Knoto-ID. Add command line option to specify path to Knoto-ID?
        s = "this will be knoto-id output"
        try:
            return KnotoidClass(s)
        except ValueError:
            return KnotoidClass.UNCLASSIFIED

    @staticmethod
    def _inverse_projection(point: np.ndarray) -> np.ndarray:
        """
        Inverse stereographic projection.
        """
        return np.array(
            [2 * point[0], 2 * point[1], -1 + np.linalg.norm(point) ** 2]
        ) / (1 + np.linalg.norm(point) ** 2)

    @staticmethod
    def _remove_leaves(
        nodes: SphericalNodeDict, edges: List[Edge]
    ) -> Tuple[SphericalNodeDict, List[Edge]]:
        """
        Removes all leaves from the graph.
        Algorithm:
            1. Find a leaf, which is a vertex with valance 1.
            2. Trim the leaf, and keep trimming along this strand until the strand is
            deleted. Note this is because removing a leaf may create a new leaf.
            3. Continue until there are no leaves.
        """
        adjacency_matrix = Grapher._edge_list_to_adjacency_matrix(edges)
        while len(np.where(np.sum(adjacency_matrix, axis=0) == 1)[1]) > 0:
            leaf = np.where(np.sum(adjacency_matrix, axis=0) == 1)[1][0]
            adjacency_matrix = Grapher._trim_leaf_strand(adjacency_matrix, leaf)
        isolated_vertices = np.where(adjacency_matrix.toarray().any(axis=1) == 0)
        for v in list(isolated_vertices[0]):
            if v in nodes:
                del nodes[v]
        return nodes, Grapher._adjacency_matrix_to_edge_list(adjacency_matrix)

    @staticmethod
    def _trim_leaf_strand(
        adjacency_matrix: sp.sparse.dok_matrix, leaf: int
    ) -> sp.sparse.dok_matrix:
        """
        Delete all edges along an entire leaf strand.
        """
        neighbour = np.argwhere(adjacency_matrix[:, leaf] != 0)[0][0]
        if np.sum(adjacency_matrix[neighbour, :]) > 2:
            adjacency_matrix[leaf, neighbour] = 0
            adjacency_matrix[neighbour, leaf] = 0
            return adjacency_matrix
        adjacency_matrix[leaf, neighbour] = 0
        adjacency_matrix[neighbour, leaf] = 0
        return Grapher._trim_leaf_strand(adjacency_matrix, neighbour)

    @staticmethod
    def _edge_list_to_adjacency_matrix(edges: List[Edge]) -> sp.sparse.dok_matrix:
        """
        Returns the adjacency matrix of the graph.
        """
        n = np.max(np.array(edges)) + 1
        x = sp.sparse.dok_matrix((n, n), dtype=int)
        for edge in edges:
            x[edge[0], edge[1]] = 1
        x += x.T
        return x

    @staticmethod
    def _adjacency_matrix_to_edge_list(
        adjacency_matrix: sp.sparse.dok_matrix,
    ) -> List[Edge]:
        """
        Convert adjacency matrix to list of edges.
        """
        indices = sp.sparse.triu(adjacency_matrix).nonzero()
        new_edges = []
        for i in range(indices[0].shape[0]):
            new_edges.append((indices[0][i], indices[1][i]))
        return new_edges


def subdivide_curve(curve: np.ndarray, factor: int) -> np.ndarray:
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


if __name__ == "__main__":
    # development only
    pl_curve = np.array(
        [
            [3.39348426095347, -0.00611687458471819, 0.000417557750258721],
            [2.96604530691374, 0.951870338215143, 0.821666924650344],
            [1.83930657084388, 1.54777726947254, 1.25626134099222],
            [0.469135475523462, 1.54406959645954, 1.00592086474954],
            [-0.60142421425641, 1.04286446588196, 0.00497601351961938],
            [-1.5713824251345, 0.368238123058527, -1.00057504682367],
            [-2.26294813388938, -0.815496650449799, -1.25471722615356],
            [-2.3128076565279, -2.09044902532547, -0.82125065999155],
            [-1.6976583796246, -2.94075078800773, 0.000564375447807511],
            [-0.653580716731356, -3.04835372484424, 0.822162818423712],
            [0.425913795097334, -2.36807978977476, 1.2551661637214],
            [1.10590405603033, -1.1779626362436, 1.00077245962767],
            [1.10466082113733, 1.17711784671551, -1.00526125896886],
            [0.422660695970157, 2.36559494992422, -1.25563265937198],
            [-0.657100033636752, 3.04313857097233, -0.821135701066607],
            [-1.70068973965882, 2.93365580297979, -8.24987795427573e-05],
            [-2.31550496188464, 2.08292085981145, 0.820674799650177],
            [-2.26564166770125, 0.808510887218917, 1.25408407933773],
            [-1.57392084030126, -0.374752151141274, 1.00146652820878],
            [-0.603055310031078, -1.04932092056333, -0.00231688946508329],
            [0.467083058443442, -1.55224514368384, -1.00414878657081],
        ]
    )
    pl_curve = subdivide_curve(pl_curve, 5)
    grapher = Grapher(pl_curve)
    regions = list(grapher.compute_regions())
    plotting.plot_spherical_regions(regions, pl_curve)
