import copy
import itertools
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import planar
import scipy as sp
from knotoid_class import KnotoidClass
from spherical_geometry import great_circle_arc, polygon
from structures import Edge, Region, SphericalNode, SphericalNodeDict

logging.basicConfig(level=logging.INFO)


# TODO: class needs a refactor; graph creation, region creation and documention should be separated
class Grapher:
    """
    Computes the boundaries of the regions on the sphere, each of which corresponds to a different knotoid classification.

    Calls Knoto-ID to classify each region.
    """

    def __init__(self, source: Path, path_to_ki: Path):
        self.source = source
        self.path_to_ki = path_to_ki
        # TODO: move this during refactor. Shouldn't be in this class.
        file_extension = self.source.suffix
        if file_extension == ".txt":
            self.pl_curve = np.loadtxt(self.source)
        elif file_extension == ".npy":
            self.pl_curve = np.load(self.source)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

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
            if face.internal_point is not None:
                internal_point = Grapher._inverse_projection(face.internal_point)
                is_external = False
            else:
                # north pole is the representative point of the external face
                internal_point = np.array([0, 0, 1])
                is_external = True
            # TODO: area not correct for external region bounded by two curves.
            area = polygon.SphericalPolygon(
                [node.position for node in boundary_nodes]
                + [boundary_nodes[0].position],
                internal_point,
            ).area() / (4 * np.pi)
            # TODO: performance cost calling Knoto-ID for each region; refactor to classify all points at once
            classification = self._classify_region(internal_point)
            yield Region(
                internal_point=internal_point,
                boundary_nodes=boundary_nodes,
                knotoid_class=classification,
                area=area,
                is_external=is_external,
            )

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
        exe_path = Path.joinpath(self.path_to_ki, "bin/polynomial_invariant")
        command = f'{exe_path} --projection="{point[0]}, {point[1]}, {point[2]}" --names-db=internal {self.source}'
        # Execute the command using subprocess and capture output
        try:
            completed_process = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calling Knoto-ID: {e}.")
            raise e
        stdout = completed_process.stdout
        s = ""
        # Parse the stdout to find the knotoid type
        for line in stdout.split("\n"):
            if "Knotoid type:" in line:
                s = line.split(":")[1].split()[0]
                break
        try:
            return KnotoidClass(s)
        except ValueError:
            return KnotoidClass.UNCLASSIFIED

    @staticmethod
    def _inverse_projection(point: np.ndarray) -> np.ndarray:
        """
        Inverse stereographic projection.
        """
        # TODO: getting acos error w/ Knoto-ID; thought it was due to floating point
        # error but not sure after normalising here
        projected_point = np.array(
            [2 * point[0], 2 * point[1], -1 + np.linalg.norm(point) ** 2]
        ) / (1 + np.linalg.norm(point) ** 2)
        return projected_point / np.linalg.norm(projected_point)

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


def summarise_distribution(regions: List[Region]) -> None:
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


if __name__ == "__main__":
    # development only
    source_path = Path("tests/data/pl_curve.txt")
    path_to_ki = Path("/Users/nayayerolemou/Desktop/Knoto-ID")
    pl_curve = subdivide_curve(np.loadtxt(source_path), 5)
    grapher = Grapher(source_path, path_to_ki=path_to_ki)
    regions = list(grapher.compute_regions())
    summarise_distribution(regions)
    # plotting.plot_planar_regions(regions)
    # plotting.plot_spherical_regions(regions, pl_curve)
