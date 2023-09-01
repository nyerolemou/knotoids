"""Generates spherical regions and computes their knotoid classification."""

import logging
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
from spherical_geometry import polygon

from .config import Config
from .graph import Face, Region, SphericalNode
from .knotoid_class import KnotoidClass


def generate_regions_from_faces(
    faces: Iterable[Face], config: Config
) -> Iterable[Region]:
    """
    Returns regions, each of which corresponds to a different knotoid classification.
    """
    for face in faces:
        boundary_nodes = [
            SphericalNode(node.index, _inverse_projection(node.position))
            for node in face.boundary_nodes
        ]
        if face.internal_point is not None:
            internal_point = _inverse_projection(face.internal_point)
            is_external = False
        else:
            # north pole is the representative point of the external face
            internal_point = np.array([0, 0, 1])
            is_external = True
        # TODO: area not correct for external region bounded by two curves.
        area = polygon.SphericalPolygon(
            [node.position for node in boundary_nodes] + [boundary_nodes[0].position],
            internal_point,
        ).area() / (4 * np.pi)
        # TODO: performance cost calling Knoto-ID for each region; refactor to classify all points at once
        classification = _classify_region(internal_point, config)
        yield Region(
            internal_point=internal_point,
            boundary_nodes=boundary_nodes,
            knotoid_class=classification,
            area=area,
            is_external=is_external,
        )


def _classify_region(point: np.ndarray, config: Config) -> KnotoidClass:
    """
    Finds the knotoid classification of each region using Knoto-ID.
    """
    exe_path = Path.joinpath(config.knoto_id_root, "bin/polynomial_invariant")
    command = f'{exe_path} --projection="{point[0]}, {point[1]}, {point[2]}" --names-db=internal {config.source}'
    # Execute the command using subprocess and capture output
    try:
        completed_process = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Unexpected error whilst calling Knoto-ID: {e}.")
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
