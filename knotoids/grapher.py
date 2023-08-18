import copy
import itertools
from collections import defaultdict
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import spherical_geometry as sg
from planar import PlanarGraph


class SphericalNode(NamedTuple):
    index: int
    position: np.array


# note: a spherical/planar edge is just a pair of nodes (i,j) specified by vertex index


# TODO: think of a better name
class Grapher:
    """
    # TODO: docstring
    """

    def __init__(self, pl_curve: np.array):
        self.pl_curve = pl_curve

    def compute_regions(self) -> None:
        """
        Returns a list of regions, each of which corresponds to a different knotoid classification.

        Each region is ???TBD???
        """
        spherical_graph = self._compute_graph()
        planar_graph = PlanarGraph(spherical_graph)
        planar_regions = planar_graph.get_regions()
        spherical_regions = self._map_planar_regions_to_sphere(planar_regions)
        return self._classify_regions(spherical_regions)

    def _compute_graph(self) -> None:
        """
        Computes graph on the surface of the sphere, whose regions correspond to a (potentially different) knotoid classification.

        Stages of algorithm:
        1. Find two connected graphs on the sphere formed from normalised vectors between points and each fixed endpoint.
        2. Find the intersection of these two graphs, adding new vertices and edges where required.
        3. Remove all leaves from the graph.

        Suppose self.pl_curve has n+1 vertices {v0, v1, ..., vn}.

        Then the two connected graphs, G1 and G2, each have:
        - 2n-1 vertices
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

        # create vertex dictionary
        nodes = {}
        for i in range(2 * n - 1):
            # add vertices from first curve
            nodes[i] = SphericalNode(index=i, position=g[i])
            # add vertices from antipodal curve
            # TODO: is the copy necessary?
            nodes[i + 2 * n - 1] = SphericalNode(
                index=i + 2 * n - 1, position=-g[i].copy()
            )
        # edges are consecutive pairs of vertices, excluding any edges that join antipodal graphs
        edges = [(i, i + 1) for i in range(2 * n - 1)] + [
            (i, i + 1) for i in range(2 * n - 1, 4 * n - 3)
        ]

        # find intersection of pair of antipodal curves
        nodes, edges = self._resolve_intersections(nodes, edges)
        # remove leaves by successively removing leaf strands
        # self._remove_leaves()

    def _resolve_intersections(
        self, nodes: Dict[int, SphericalNode], edges: List[Tuple[int, int]]
    ) -> Tuple[Dict[int, SphericalNode], List[Tuple[int, int]]]:
        # TODO: should edges be a set instead of a list?
        """
        Finds the intersection of two graphs, adding new vertices and edges where required.

        Credit to Alex for the algorithm.
        """
        nodes = copy.copy(nodes)
        edges = copy.copy(edges)

        intersection_tracker = defaultdict(list)
        # note: O(n^2) - can we do better?
        for first_edge, second_edge in itertools.combinations(edges, 2):
            intersection = sg.great_circle_arc.intersection(
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
            new_node_distances_to_first_vertex = [
                np.arccos(np.dot(nodes[edge[0]].position, nodes[new_node].position))
                for new_node in new_nodes
            ]
            new_nodes = [
                new_node
                for _, new_node in sorted(
                    zip(new_node_distances_to_first_vertex, new_nodes)
                )
            ]
            # add new edges
            edges.extend(
                [(new_nodes[i], new_nodes[i + 1]) for i in range(len(new_nodes) - 1)]
            )
            # remove old edge
            edges.remove(edge)
        return nodes, edges

    # def _remove_leaves(self) -> None:
    #     """
    #     Removes all leaves from the graph.
    #     """
    #     adjacency_matrix = utils.adjacency_matrix(edges)

    #     while len(np.where(np.sum(adjacency_matrix, axis=0) == 1)[1]) > 0:
    #         leaf = np.where(np.sum(adjacency_matrix, axis=0) == 1)[1][0]
    #         adjacency_matrix = self._trim_leaf_strand(adjacency_matrix, leaf)

    #     return adjacency_matrix

    # def _trim_leaf_strand(self, adj, leaf):
    #     """
    #     Params:
    #         adj -- sparse adjacency matrix
    #         leaf -- leaf vertex
    #     Returns:
    #         matrix with leaf strand starting from v removed
    #     """
    #     nbr = np.argwhere(adj[:, leaf] != 0)[0][0]

    #     if np.sum(adj[nbr, :]) > 2:
    #         adj[leaf, nbr] = 0
    #         adj[nbr, leaf] = 0

    #         return adj

    #     adj[leaf, nbr] = 0
    #     adj[nbr, leaf] = 0

    #     return self._trim_leaf_strand(adj, nbr)

    def _classify_regions(self) -> None:
        """
        Finds the knotoid classification of each region using Knoto-ID.
        """
        pass

    # @classmethod
    # def _get_adjacency_matrix(edges: Set[Tuple[int, int]]) -> :
    #     """
    #     Returns the adjacency matrix of the graph.
    #     """
    #     pass
