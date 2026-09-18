#!/urs/bin/env python3


import logging
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from polarity_engine.builder import ProteinGraphBuilder, DEFAULT_MAX_ASA
from tests.mock_data import (
    MOCK_NODES,
    MOCK_SASA_MAP,
    MOCK_AA_LIST,
    MOCK_COORDS_NP,
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
        MOCK_PROTEIN_5RES["nodes"],
        MOCK_PROTEIN_5RES["sasa_map"],
        MOCK_PROTEIN_5RES["name"],
    )


@pytest.fixture
def dummy_single_residue_protein():
    """
    Returns a single residue synthetic protein.
    """
    return (
        MOCK_PROTEIN_SINGLE_RES["aa_list"],
        MOCK_PROTEIN_SINGLE_RES["coords"],
        MOCK_PROTEIN_SINGLE_RES["nodes"],
        MOCK_PROTEIN_SINGLE_RES["sasa_map"],
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
        MOCK_PROTEIN_CORRUPTED_NAN["nodes"],
        MOCK_PROTEIN_CORRUPTED_NAN["sasa_map"],
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
        MOCK_PROTEIN_CORRUPTED_INF["nodes"],
        MOCK_PROTEIN_CORRUPTED_INF["sasa_map"],
        MOCK_PROTEIN_CORRUPTED_INF["name"],
    )


class TestProteinGraphBuilder:
    """
    Groups all unit tests validating the state and side-effects of ProteinGraphBuilder
    """

    def test_tensor_shapes_and_types(self, builder, dummy_5_residue_protein):
        """Validates node, edge, and coordinate tensor dimensions.

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence,
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verify output tensors match expected shape contracts and torch dtypes.
        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein
        data = builder.build_graph(
            aa_list=aa_list, coords_np=coords, nodes=nodes, sasa_map=sasa_map, name=name
        )

        N = len(aa_list)
        E = data.edge_index.shape[1]

        # Node Features: (N, 23) -> 21 one-hot + 1 sequence position + 1 rSASA
        assert data.x.shape == (N, 23)
        assert data.x.dtype == torch.float32

        # Coordinates: (N, 3)
        assert data.pos.shape == (N, 3)
        assert data.pos.dtype == torch.float32

        # Edge Index: (2, E)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.dtype == torch.int64

        # Edge Attributes: (E, 20) -> 3 unit vec + 16 RBF + 1 inter-chain flag
        assert data.edge_attr.shape == (E, 20)
        assert data.edge_attr.dtype == torch.float32

    def test_no_divide_by_zero(self, builder, dummy_single_residue_protein):
        """Verifies sequence position normalization prevents division-by-zero for single-residue inputs (N=1).

        Arrange:
            dummy_single_residue_protein fixture providing 1-residue sequence,
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder on N=1 complex.
        Assert:
            Verify output node features contain no NaNs, sequence position
            defaults to 0.0, and rSASA is computed properly.
        """
        aa_list, coords, nodes, sasa_map, name = dummy_single_residue_protein
        data = builder.build_graph(
            aa_list=aa_list,
            coords_np=coords,
            nodes=nodes,
            sasa_map=sasa_map,
            name=name,
        )

        # 1. Verify tensor shapes
        assert data.x.shape == (1, 23)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 19)

        # 2. Core Zero-Division & Bounds Validation
        assert not torch.isnan(
            data.x
        ).any(), "Node features contain NaN from division by zero!"

        # Explicitly extract feature columns for clarity
        # Index -2: Normalized sequence position i / max(N-1, 1)
        seq_pos_scalar = data.x[0, -2].item()
        # Index -1: Relative SASA in [0.0, 1.0]
        rsasa_scalar = data.x[0, -1].item()

        assert (
            seq_pos_scalar == 0.0
        ), "Single-residue sequence position should normalize to 0.0"
        assert (
            0.0 <= rsasa_scalar <= 1.0
        ), "Single-residue rSASA must be bounded in [0.0, 1.0]"

    def test_corrupt_protein_nan(self, builder, corrupted_protein_nan):
        """
        Verify that ValueError is raised when coordinates contain NaN values.

        Arrange:
            corrupted_protein_nan fixture providing coordinates with NaN.
        Act & Assert:
            Invoke build_graph and catch ValueError with expected message context.
        """
        aa_list, coords, nodes, sasa_map, name = corrupted_protein_nan
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

    def test_corrupt_protein_inf(self, builder, corrupted_protein_inf):
        """
        Verify that ValueError is raised when coordinates contain Inf values.

        Arrange:
            corrupted_protein_inf fixture providing coordinates with Inf.
        Act & Assert:
            Invoke build_graph and catch ValueError with expected message context.
        """
        aa_list, coords, nodes, sasa_map, name = corrupted_protein_inf
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

    def test_corrupt_protein_nodes_mismatch(self, builder, dummy_5_residue_protein):
        """
        Verify that ValueError is raised when nodes metadata length mismatches aa_list length.

        Arrange:
            dummy_single_residue_protein fixture providing 1-residue sequence,
            coordinates, node metadata, and SASA map, and truncate nodes to
            simulate a missing metadata entry.
        Act & Assert:
            Invoke build_graph and catch ValueError with expected message context.
        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein

        # Truncate nodes to simulate a missing metadata entry (len = 4 vs len = 5)
        truncated_nodes = nodes[:-1]

        with pytest.raises(ValueError, match="Length mismatch in structure"):
            builder.build_graph(
                aa_list=aa_list,
                coords_np=coords,
                nodes=truncated_nodes,
                sasa_map=sasa_map,
                name=name,
            )

    def test_unit_vectors_normalized(self, builder, dummy_5_residue_protein):
        """
        Verifies direction vectors in edge_attr[:, :3] have unit norm (~1.0).

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence,
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Direction vectors in edge_attr[:, :3] have unit norm (~1.0)

        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

        unit_vecs = data.edge_attr[:, :3]
        norms = torch.linalg.vector_norm(unit_vecs, dim=-1)

        # Unit vector lengths should equal 1.0 within floating point precision
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_sequence_position_normalization(self, builder, dummy_5_residue_protein):
        """
        Verifies sequence position scalar stays strictly bounded in [0.0, 1.0].

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verifies sequence position scalar stays strictly bounded in [0.0, 1.0].
        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

        # Sequence position is now column index -2 (index 21 out of 23)
        seq_positions = data.x[:, -2]
        assert seq_positions[0].item() == 0.0
        assert seq_positions[-1].item() == 1.0
        assert (seq_positions >= 0.0).all() and (seq_positions <= 1.0).all()

    def test_graph_symmetry_and_topology(self, builder, dummy_5_residue_protein):
        """
        Validates distance cutoff filtering and edge symmetry.

        Arrange:
            dummy_5_residue_protein fixture providing 5-residue sequence and
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Validates distance cutoff filtering and edge symmetry.
        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

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
            coordinates, node metadata, and SASA map.
        Act:
            Invoke build_graph of ProteinGraphBuilder to construct PyG Data object.
        Assert:
            Verify for every edge (i -> j), the reverse edge (j -> i) exists.
        """
        aa_list, coords, nodes, sasa_map, name = dummy_5_residue_protein
        data = builder.build_graph(aa_list, coords, nodes, sasa_map, name=name)

        edge_index = data.edge_index
        # Transpose edges to set of tuples (i, j)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        # Assert symmetry: if (i, j) in edges, then (j, i) must be in edges
        for src, dst in edges:
            assert (dst, src) in edges, f"Edge ({src} -> {dst}) lacks reverse edge!"

    def test_build_rsasa_node_tensor_from_mock_data(self):
        """
        Directly tests build_rsasa_node_tensor using standardized mock data.

        Arrange:
            Import MOCK_NODES and MOCK_SASA_MAP containing standard, core-buried,
            non-standard fallback, and out-of-bounds SASA test cases.
        Act:
            Call ProteinGraphBuilder.build_rsasa_node_tensor static method.
        Assert:
            Verify output tensor shape is (4, 1), dtype is float32, expected relative
            SASA values match calculations, and values are clamped within [0.0, 1.0].
        """
        tensor = ProteinGraphBuilder.build_rsasa_node_tensor(MOCK_NODES, MOCK_SASA_MAP)

        # Shape and dtype assertions
        assert tensor.shape == (4, 1)
        assert tensor.dtype == torch.float32

        # Value correctness based on mock data values
        assert torch.isclose(tensor[0], torch.tensor([0.5]))
        assert torch.isclose(tensor[1], torch.tensor([0.0]))
        assert torch.isclose(tensor[2], torch.tensor([0.5]))
        assert torch.isclose(tensor[3], torch.tensor([1.0]))  # Clamped bound

        # Bound Range Check
        assert (tensor >= 0.0).all() and (tensor <= 1.0).all()

    def test_build_graph_with_mock_rsasa(self, builder):
        """
        Verifies build_graph outputs (N, 23) tensor using imported mock data.

        Arrange:
            Prepare 4-residue coordinate array, node metadata, and SASA map from standardized mock data.
        Act:
            Invoke build_graph to generate PyG Data instance.
        Assert:
            Verify node feature matrix x has shape (4, 23) and that column index -1 exactly matches
            the expected rSASA values [0.5, 0.0, 0.5, 1.0].
        """
        coords_np = np.array(MOCK_COORDS_NP, dtype=np.float32)

        data = builder.build_graph(
            aa_list=MOCK_AA_LIST,
            coords_np=coords_np,
            nodes=MOCK_NODES,
            sasa_map=MOCK_SASA_MAP,
            name="mock_complex",
        )

        # Validate output tensor dimension (22 base features + 1 rSASA = 23)
        assert data.x.shape == (4, 23)

        # Verify that the last feature column corresponds to the computed rSASA values
        rsasa_column = data.x[:, -1]
        assert torch.allclose(rsasa_column, torch.tensor([0.5, 0.0, 0.5, 1.0]))
