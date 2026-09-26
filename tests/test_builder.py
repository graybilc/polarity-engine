#!/usr/bin/env python3

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from polarity_engine.builder import ProteinGraphBuilder
from tests.mock_data import (
    MOCK_NODES,
    MOCK_AA_LIST,
    MOCK_COORDS_NP,
    MOCK_B_FACTORS_NP,
    MOCK_OCCUPANCIES_NP,
    MOCK_RSASA_NP,
    MOCK_PROTEIN_5RES,
    MOCK_PROTEIN_CORRUPTED_NAN,
    MOCK_PROTEIN_SINGLE_RES,
    MOCK_PROTEIN_CORRUPTED_INF,
)

SEQ_POS_COL = 21
RSASA_COL = 22
B_FACTOR_COL = 23
OCCUPANCY_COL = 24


@pytest.fixture
def builder():
    """
    Fixture for builder with standard 8.0 A cutoff and 16 RBF kernels.
    """
    return ProteinGraphBuilder(distance_cutoff=8.0, num_rbf_kernels=16)


@pytest.fixture
def dummy_5_residue_protein():
    """
    Returns a standard 5-residue synthetic protein tuple.
    """
    return (
        MOCK_PROTEIN_5RES["aa_list"],
        MOCK_PROTEIN_5RES["coords"],
        MOCK_PROTEIN_5RES["b_factors"],
        MOCK_PROTEIN_5RES["occupancies"],
        MOCK_PROTEIN_5RES["nodes"],
        MOCK_PROTEIN_5RES["rsasa"],
        MOCK_PROTEIN_5RES["name"],
    )


@pytest.fixture
def dummy_single_residue_protein():
    """
    Returns a single residue synthetic protein tuple.
    """
    return (
        MOCK_PROTEIN_SINGLE_RES["aa_list"],
        MOCK_PROTEIN_SINGLE_RES["coords"],
        MOCK_PROTEIN_SINGLE_RES["b_factors"],
        MOCK_PROTEIN_SINGLE_RES["occupancies"],
        MOCK_PROTEIN_SINGLE_RES["nodes"],
        MOCK_PROTEIN_SINGLE_RES["rsasa"],
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
        MOCK_PROTEIN_CORRUPTED_NAN["b_factors"],
        MOCK_PROTEIN_CORRUPTED_NAN["occupancies"],
        MOCK_PROTEIN_CORRUPTED_NAN["nodes"],
        MOCK_PROTEIN_CORRUPTED_NAN["rsasa"],
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
        MOCK_PROTEIN_CORRUPTED_INF["b_factors"],
        MOCK_PROTEIN_CORRUPTED_INF["occupancies"],
        MOCK_PROTEIN_CORRUPTED_INF["nodes"],
        MOCK_PROTEIN_CORRUPTED_INF["rsasa"],
        MOCK_PROTEIN_CORRUPTED_INF["name"],
    )


class TestProteinGraphBuilder:
    """
    Groups all unit tests validating state and tensor outputs of ProteinGraphBuilder.
    """

    def test_tensor_shapes_and_types(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Extract standard 5-residue protein mock data.
        Act:
            Build PyTorch Geometric graph via ProteinGraphBuilder.
        Assert:
            Validate node, edge, coordinate, and attribute tensor shapes/dtypes.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )
        data = builder.build_graph(
            aa_list=aa_list,
            coords_np=coords,
            b_factors_np=b_factors,
            occupancies_np=occupancies,
            nodes=nodes,
            rsasa_np=rsasa,
            name=name,
        )

        N = len(aa_list)
        E = data.edge_index.shape[1]

        # Node Features: (N, 25) -> 21 one-hot + 1 seq_pos + 1 rSASA + 1 B-factor + 1 occupancy
        assert data.x.shape == (N, 25)
        assert data.x.dtype == torch.float32

        # Verify rSASA, B-factors, and Occupancies in node feature tensor slices
        assert torch.allclose(data.x[:, RSASA_COL], torch.from_numpy(rsasa).float())
        assert torch.allclose(
            data.x[:, B_FACTOR_COL], torch.from_numpy(b_factors).float()
        )
        assert torch.allclose(
            data.x[:, OCCUPANCY_COL], torch.from_numpy(occupancies).float()
        )

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
        """
        Arrange:
            Extract single-residue synthetic protein input (N=1).
        Act:
            Construct graph representation.
        Assert:
            Verify zero-division prevention for single sequence positions and feature bounds.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_single_residue_protein
        )
        data = builder.build_graph(
            aa_list=aa_list,
            coords_np=coords,
            b_factors_np=b_factors,
            occupancies_np=occupancies,
            nodes=nodes,
            rsasa_np=rsasa,
            name=name,
        )

        # 1. Verify tensor shapes
        assert data.x.shape == (1, 25)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 20)

        # 2. Core Zero-Division & Bounds Validation
        assert not torch.isnan(
            data.x
        ).any(), "Node features contain NaN from division by zero!"

        seq_pos_scalar = data.x[0, SEQ_POS_COL].item()
        rsasa_scalar = data.x[0, RSASA_COL].item()

        assert (
            seq_pos_scalar == 0.0
        ), "Single-residue sequence position should normalize to 0.0"
        assert (
            0.0 <= rsasa_scalar <= 1.0
        ), "Single-residue rSASA must be bounded in [0.0, 1.0]"

    def test_corrupt_protein_nan(self, builder, corrupted_protein_nan):
        """
        Arrange:
            Extract protein dataset containing NaN coordinates.
        Act:
            Attempt building graph structure.
        Assert:
            Expect ValueError due to invalid coordinate numerical safety checks.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            corrupted_protein_nan
        )
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(
                aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
            )

    def test_corrupt_protein_inf(self, builder, corrupted_protein_inf):
        """
        Arrange:
            Extract protein dataset containing Inf coordinates.
        Act:
            Attempt building graph structure.
        Assert:
            Expect ValueError due to infinite coordinate values.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            corrupted_protein_inf
        )
        with pytest.raises(ValueError, match="Invalid coordinates in structure"):
            builder.build_graph(
                aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
            )

    def test_corrupt_protein_nodes_mismatch(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Extract 5-residue protein and truncate nodes metadata to length 4.
        Act:
            Attempt graph generation with mismatched length vectors.
        Assert:
            Expect ValueError flagging metadata length mismatch.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )

        # Truncate nodes to simulate a missing metadata entry (len = 4 vs len = 5)
        truncated_nodes = nodes[:-1]

        with pytest.raises(ValueError, match="Length mismatch in structure"):
            builder.build_graph(
                aa_list=aa_list,
                coords_np=coords,
                b_factors_np=b_factors,
                occupancies_np=occupancies,
                nodes=truncated_nodes,
                rsasa_np=rsasa,
                name=name,
            )

    def test_unit_vectors_normalized(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Build graph from standard 5-residue mock dataset.
        Act:
            Slice unit direction vectors from edge attribute tensor.
        Assert:
            Validate vector norms equal 1.0 within floating point tolerance.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )
        data = builder.build_graph(
            aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
        )

        unit_vecs = data.edge_attr[:, :3]
        norms = torch.linalg.vector_norm(unit_vecs, dim=-1)

        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_sequence_position_normalization(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Generate graph from 5-residue protein.
        Act:
            Extract relative sequence position column.
        Assert:
            Confirm values lie strictly within range [0.0, 1.0] from first to last residue.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )
        data = builder.build_graph(
            aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
        )

        seq_positions = data.x[:, SEQ_POS_COL]
        assert seq_positions[0].item() == 0.0
        assert seq_positions[-1].item() == 1.0
        assert (seq_positions >= 0.0).all() and (seq_positions <= 1.0).all()

    def test_graph_symmetry_and_topology(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Generate graph from 5-residue protein with an isolated node beyond distance cutoff.
        Act:
            Extract edge index and filter inter-node connections.
        Assert:
            Confirm distant node (index 4) remains unconnected.
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )
        data = builder.build_graph(
            aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
        )

        edge_index = data.edge_index
        inter_node_mask = edge_index[0] != edge_index[1]
        inter_edge_index = edge_index[:, inter_node_mask]

        connected_nodes = torch.unique(inter_edge_index)

        # Node 4 (at 15.0 Å) has no neighbors within the 8.0 Å cutoff
        assert (
            4 not in connected_nodes
        ), "Isolated node beyond distance cutoff was incorrectly connected to another node"

    def test_edge_index_is_symmetric(self, builder, dummy_5_residue_protein):
        """
        Arrange:
            Build protein graph.
        Act:
            Collect unique directed edge index tuples.
        Assert:
            Ensure bidirectional presence of every edge pair (i -> j) and (j -> i).
        """
        aa_list, coords, b_factors, occupancies, nodes, rsasa, name = (
            dummy_5_residue_protein
        )
        data = builder.build_graph(
            aa_list, coords, b_factors, occupancies, nodes, rsasa, name=name
        )

        edge_index = data.edge_index
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        for src, dst in edges:
            assert (dst, src) in edges, f"Edge ({src} -> {dst}) lacks reverse edge!"

    def test_build_graph_with_mock_rsasa(self, builder):
        """
        Arrange:
            Load standard direct NumPy mock fixtures.
        Act:
            Build graph with direct mock array injection.
        Assert:
            Verify output node shapes and matching rSASA column slices.
        """
        coords_np = np.array(MOCK_COORDS_NP, dtype=np.float32)

        data = builder.build_graph(
            aa_list=MOCK_AA_LIST,
            coords_np=coords_np,
            b_factors_np=MOCK_B_FACTORS_NP,
            occupancies_np=MOCK_OCCUPANCIES_NP,
            nodes=MOCK_NODES,
            rsasa_np=MOCK_RSASA_NP,
            name="mock_complex",
        )

        assert data.x.shape == (4, 25)
        rsasa_column = data.x[:, RSASA_COL]
        assert torch.allclose(rsasa_column, torch.from_numpy(MOCK_RSASA_NP).float())
