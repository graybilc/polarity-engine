#!/urs/bin/env python3


import logging
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from polarity_engine.builder import GaussianRBF, ProteinGraphBuilder
from tests.mock_data import (
    MOCK_PROTEIN_5RES,
    MOCK_PROTEIN_CORRUPTED_NAN,
    MOCK_PROTEIN_SINGLE_RES,
    MOCK_PROTEIN_CORRUPTED_INF,
)


@pytest.fixture
def builder():
    """
    Fixture for builder with standard 8.0 A cutoff and 16 RBF kernels.
    """
    return ProteinGraphBuilder(distance_cutoff=8.0, num_rbf_kernels=16)


@pytest.fixture
def dummy_5_residue_protein():
    """
    Returns a standard 5-residue synthetic protein.
    """
    return (
        MOCK_PROTEIN_5RES["aa_list"],
        MOCK_PROTEIN_5RES["coords"],
        MOCK_PROTEIN_5RES["name"],
    )


@pytest.fixture
def dummy_single_resisude_protein():
    """
    Returns a single residue synthetic protein.
    """
    return (
        MOCK_PROTEIN_SINGLE_RES["aa_list"],
        MOCK_PROTEIN_SINGLE_RES["coords"],
        MOCK_PROTEIN_SINGLE_RES["name"],
    )


@pytest.fixture
def corrupted_protein_nan():
    """
    Returns a protein complex containing NaN coordinates.
    """
    return (
        MOCK_PROTEIN_CORRUPTED_NAN["aa_list"],
        MOCK_PROTEIN_CORRUPTED_NAN["coords"],
        MOCK_PROTEIN_CORRUPTED_NAN["name"],
    )


@pytest.fixture
def corrupted_protein_inf():
    """
    Returns a protein complex containing Inf coordinates.
    """
    return (
        MOCK_PROTEIN_CORRUPTED_INF["aa_list"],
        MOCK_PROTEIN_CORRUPTED_INF["coords"],
        MOCK_PROTEIN_CORRUPTED_INF["name"],
    )


class TestProteinGraphBuilder:
    """
    Groups all unit tests validating the state and side-effects of ProteinGraphBuilder
    """

    def test_tensor_shapes_and_types(self, builder, dummy_5_residue_protein):
        """
        Validates node, edge, and coordinate tensor dimensions.

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verify output tensors match expected shape contracts and torch dtypes.
        """
        aa_list, coords, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, name=name)

        N = len(aa_list)
        E = data.edge_index.shape[1]

        # Node Features: (N, 22) -> 21 one-hot + 1 position
        assert data.x.shape == (N, 22)
        assert data.x.dtype == torch.float32

        # Coordinates: (N, 3)
        assert data.pos.shape == (N, 3)
        assert data.pos.dtype == torch.float32

        # Edge Index: (2, E)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.dtype == torch.int64

        # Edge Attributes: (E, 19) -> 3 unit vec + 16 RBF
        assert data.edge_attr.shape == (E, 19)
        assert data.edge_attr.dtype == torch.float32

    def test_no_divide_by_zero(self, builder, dummy_single_resisude_protein):
        """
        Verifies sequence position normalization prevents division-by-zero for single-residue inputs (N=1).

        Arrange:
            dummy_single_resisude_protein fixture providing 1-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder on N=1 complex.
        Assert:
            Verify output node features contain no NaNs and sequence position
            defaults to 0.0.
        """
        aa_list, coords, name = dummy_single_resisude_protein
        data = builder.build_graph(aa_list, coords, name=name)

        # Verify tensor shapes
        assert data.x.shape == (1, 22)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 19)

        # Core Zero-Division Validation
        assert not torch.isnan(
            data.x
        ).any(), "Node features contain NaN from division by zero!"
        assert (
            data.x[0, -1].item() == 0.0
        ), "Single-residue sequence position should normalize to 0.0"

    def test_corrupt_protein_nan(self, builder, corrupted_protein_nan):
        """
        Verify that ValueError is raised when coordinates contain NaN values.

        Arrange:
            corrupted_protein_nan fixture providing coordinates with NaN.
        Act & Assert:
            Invoke build_graph and catch ValueError with expected message context.
        """
        aa_list, coords, name = corrupted_protein_nan
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(aa_list, coords, name=name)

    def test_corrupt_protein_inf(self, builder, corrupted_protein_inf):
        """
        Verify that ValueError is raised when coordinates contain Inf values.

        Arrange:
            corrupted_protein_inf fixture providing coordinates with Inf.
        Act & Assert:
            Invoke build_graph and catch ValueError with expected message context.
        """
        aa_list, coords, name = corrupted_protein_inf
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(aa_list, coords, name=name)

    def test_unit_vectors_normalized(self, builder, dummy_5_residue_protein):
        """
        Verifies direction vectors in edge_attr[:, :3] have unit norm (~1.0).

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Direction vectors in edge_attr[:, :3] have unit norm (~1.0)

        """
        aa_list, coords, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, name=name)

        unit_vecs = data.edge_attr[:, :3]
        norms = torch.linalg.vector_norm(unit_vecs, dim=-1)

        # Unit vector lengths should equal 1.0 within floating point precision
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_sequence_position_normalization(self, builder, dummy_5_residue_protein):
        """
        Verifies sequence position scalar stays strictly bounded in [0.0, 1.0].

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verifies sequence position scalar stays strictly bounded in [0.0, 1.0].
        """
        aa_list, coords, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, name=name)

        seq_positions = data.x[:, -1]
        assert seq_positions[0].item() == 0.0
        assert seq_positions[-1].item() == 1.0
        assert (seq_positions >= 0.0).all() and (seq_positions <= 1.0).all()

    def test_graph_symmetry_and_topology(self, builder, dummy_5_residue_protein):
        """
        Validates distance cutoff filtering and edge symmetry.

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Validates distance cutoff filtering and edge symmetry.
        """
        aa_list, coords, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, name=name)

        # Filter out self-loops (i == j) to inspect inter-node connectivity
        edge_index = data.edge_index
        inter_node_mask = edge_index[0] != edge_index[1]
        inter_edge_index = edge_index[:, inter_node_mask]

        connected_nodes = torch.unique(inter_edge_index)

        # Node 4 (at 15.0 Å) has no neighbors within the 8.0 Å cutoff
        assert 4 not in connected_nodes, (
            "Isolated node beyond distance cutoff was incorrectly connected to another"
            " node"
        )

    def test_edge_index_is_symmetric(self, builder, dummy_5_residue_protein):
        """
        Verifies that for every edge (i -> j), the reverse edge (j -> i) exists.

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verify for every edge (i -> j), the reverse edge (j -> i) exists.
        """
        aa_list, coords, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, name=name)

        edge_index = data.edge_index
        # Transpose edges to set of tuples (i, j)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        # Assert symmetry: if (i, j) in edges, then (j, i) must be in edges
        for src, dst in edges:
            assert (dst, src) in edges, f"Edge ({src} -> {dst}) lacks reverse edge!"
