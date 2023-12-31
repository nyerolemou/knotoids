# Generated by CodiumAI
import numpy as np
import pytest

from knotoids.graph import Region, SphericalNode
from knotoids.knotoid_class import KnotoidClass


class TestRegion:
    # Tests that a Region object can be created with internal_point, boundary_nodes, knotoid_class, and area attributes
    def test_create_region_object(self):
        internal_point = np.array([1, 0, 0])
        boundary_nodes = [
            SphericalNode(0, np.array([1, 0, 0])),
            SphericalNode(1, np.array([0, 1, 0])),
            SphericalNode(2, np.array([0, 0, 1])),
        ]
        knotoid_class = KnotoidClass.CLASS_0_1
        area = 1.0

        region = Region(internal_point, boundary_nodes, knotoid_class, area)

        np.testing.assert_array_equal(region.internal_point, internal_point)
        # assert list of boundary_nodes is equal to list of boundary_nodes
        assert len(region.boundary_nodes) == len(boundary_nodes)
        for node, boundary_node in zip(region.boundary_nodes, boundary_nodes):
            assert node.index == boundary_node.index
            np.testing.assert_array_equal(node.position, boundary_node.position)
        assert region.knotoid_class == knotoid_class
        assert region.area == area

    # Tests that a Region object can be created with is_external set to True
    def test_create_region_object_with_is_external(self):
        internal_point = np.array([0, 0, 0])
        boundary_nodes = [
            SphericalNode(0, np.array([1, 0, 0])),
            SphericalNode(1, np.array([0, 1, 0])),
            SphericalNode(2, np.array([0, 0, 1])),
        ]
        knotoid_class = KnotoidClass.CLASS_0_1
        area = 1.0
        is_external = True

        region = Region(
            internal_point, boundary_nodes, knotoid_class, area, is_external=is_external
        )

        assert region.is_external == is_external

    # Tests that a Region object can be created with an empty boundary_nodes list
    def test_create_region_object_with_empty_boundary_nodes(self):
        internal_point = np.array([0, 0, 0])
        boundary_nodes = []
        knotoid_class = KnotoidClass.CLASS_0_1
        area = 1.0

        region = Region(internal_point, boundary_nodes, knotoid_class, area)

        assert region.boundary_nodes == boundary_nodes

    # Raises ValueError if internal_point does not have unit norm.
    def test_internal_point_no_unit_norm(self):
        internal_point = np.array([2, 0, 0])
        boundary_nodes = [
            SphericalNode(0, np.array([1, 0, 0])),
            SphericalNode(1, np.array([0, 1, 0])),
        ]
        knotoid_class = KnotoidClass.CLASS_0_1
        area = 1.0
        with pytest.raises(ValueError):
            region = Region(internal_point, boundary_nodes, knotoid_class, area)

    # Tests that a Region object cannot be created with a knotoid_class not in the KnotoidClass enum
    def test_create_region_object_with_invalid_knotoid_class(self):
        internal_point = np.array([0, 0, 0])
        boundary_nodes = [
            SphericalNode(0, np.array([1, 0, 0])),
            SphericalNode(1, np.array([0, 1, 0])),
            SphericalNode(2, np.array([0, 0, 1])),
        ]
        knotoid_class = "invalid_class"
        area = 1.0

        with pytest.raises(ValueError):
            region = Region(internal_point, boundary_nodes, knotoid_class, area)
