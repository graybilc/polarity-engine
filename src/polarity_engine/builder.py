#!/urs/bin/env python3


import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Configure log
logger = logging.getLogger(__name__)

AMINO_ACID_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,  # Unknown / Non-standard
}

# Maximum Accessible Surface Area (maxASA) in square Angstroms (A^2)
# Reference: Tien et al. (2013), PLoS ONE 8(11): e80635. (Theoretical scale)
TIEN_MAX_ASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLU": 223.0,
    "GLN": 225.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

# Empirical average across standard 20 residues to handle unknown/non-standard AA classes
DEFAULT_MAX_ASA = 197.0


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function expansion module in pure PyTorch.

    Expands 1D scalar distances into a higher-dimensional continuous vector
    representation using uniformly spaced Gaussian kernels. Avoids external C++
    bindings or model-internal dependencies for maximum portability and GPU
    safety.

    Attributes:
       mu (torch.Tensor): Registered buffer containing kernel centers of shape
        (num_gaussians,).
       gamma (torch.Tensor): Registered scalar buffer controlling kernel width
        precision (1 / delta^2).
    """

    def __init__(self, start: float = 0.0, stop: float = 8.0, num_gaussians: int = 16):
        """
        Initializes kernel centers and width parameters.

        Args:
        start: Lower spatial distance bound in Ångströms (typically 0.0).
        stop: Upper spatial distance cutoff in Ångströms (e.g., 8.0).
        num_gaussians: Number of radial basis functions (K channels).
        """
        super().__init__()
        # Spaced centers (mu) from start to stop
        mu = torch.linspace(start, stop, num_gaussians)
        # Step size between centers
        delta = (stop - start) / (num_gaussians - 1)
        # Gamma (width parameter) based on step size
        gamma = 1.0 / (delta**2)

        # Register buffers so they move to GPU automatically with module
        self.register_buffer("mu", mu)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Expands scalar edge distances into Gaussian RBF feature vectors.

        Args:
            dist: Pairwise scalar distance tensor of shape (E,) or (E, 1).

        Returns:
            Expanded distance feature matrix of shape (E, num_gaussians).
        """
        # Ensure shape is (E, 1)
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)

        # Broadcast against centers: (E, 1) - (1, K) -> (E, K)
        diff = dist - self.mu.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class ProteinGraphBuilder:
    """End-to-end pipeline for constructing PyTorch Geometric Data objects from C-alpha coordinates.

    Attributes:
        distance_cutoff (float): Maximum spatial interaction distance threshold
            in Ångströms.
        rbf_module (GaussianRBF): Pure PyTorch radial basis function module for
            distance expansion.
    """

    def __init__(self, distance_cutoff: float = 8.0, num_rbf_kernels: int = 16) -> None:
        """
        Initializes builder with geometric interaction cutoffs and kernel configurations.

        Args:
            distance_cutoff: Spatial distance threshold in Ångströms (default: 8.0).
            num_rbf_kernels: Number of Gaussian channels for distance expansion
                (default: 16).
        """
        self.distance_cutoff = distance_cutoff
        self.rbf_module = GaussianRBF(
            start=0.0, stop=distance_cutoff, num_gaussians=num_rbf_kernels
        )

    def _encode_node_feature(self, aa_list: list[str]) -> torch.Tensor:
        """
        Encodes 21-dim one-hot amino acid identity and 1-dim normalized sequence position.

        Args:
            aa_list: Sequence of 3-letter amino acid code strings (e.g., ['ALA',
            'GLY', 'SER']). Non-standard codes default to 'UNK'.

        Returns:
            torch.Tensor: Float32 node feature matrix of shape (N, 22), where N =
            len(aa_list).
        """
        # Map residue strings to integer indices safely
        aa_indices = torch.tensor(
            [
                AMINO_ACID_TO_INDEX.get(res.upper(), AMINO_ACID_TO_INDEX["UNK"])
                for res in aa_list
            ],
            dtype=torch.long,
        )

        # One-hot encode to float32 tensor (N, 21)
        x_one_hot = torch.nn.functional.one_hot(aa_indices, num_classes=21).float()

        # Create normalized sequence positions in range [0, 1]
        seq_pos = torch.arange(len(aa_list), dtype=torch.float32) / max(
            len(aa_list) - 1, 1
        )
        seq_pos_col = seq_pos.unsqueeze(1)

        return torch.cat([x_one_hot, seq_pos_col], dim=1)

    def _distance_to_sparse_coo(
        self, dist_matrix: torch.Tensor, include_self_loops: bool = False
    ) -> tuple[torch.Tensor, int]:
        """
        Converts a dense pairwise distance matrix to PyTorch Geometric COO edge_index format.

        Args:
            dist_matrix: Pairwise distance tensor of shape (N, N).
            include_self_loops: Whether diagonal self-connections (i == j) are
                retained.

        Returns:
            tuple[torch.Tensor, int]:
                - edge_index: Long tensor of shape (2, E) containing source and target
                    indices.
                - num_edges: Total number of active edges E.
        """
        if include_self_loops:
            mask = dist_matrix <= self.distance_cutoff
        else:
            mask = (dist_matrix <= self.distance_cutoff) & (dist_matrix > 0.0)

        edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous().long()
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            logger.warning(
                f"No edges constructed within cutoff {self.distance_cutoff} Å."
            )
        return edge_index, num_edges

    def _compute_edge_geometric_attrs(
        self, coords: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """Computes sparse COO topology, displacement vectors, scalar distances, and unit direction vectors.

        Args:
            coords: NumPy array of C-alpha coordinates of shape (N, 3).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - 'edge_index': Long tensor of shape (2, E)
                - 'dist_scalar': Float32 distance tensor of shape (E, 1)
                - 'unit_vec': Float32 unit directional vector of shape (E, 3)
        """
        # Convert to float32 tensor
        coords_torch = torch.from_numpy(coords).float()

        # Extract sparse edge topology
        dist_matrix = torch.cdist(coords_torch, coords_torch)
        edge_index, _ = self._distance_to_sparse_coo(dist_matrix)

        # Vectorized coordinate indexing
        pos_i = coords_torch[edge_index[0]]  # (E, 3)
        pos_j = coords_torch[edge_index[1]]  # (E, 3)

        # Geometric features
        disp_vec = pos_j - pos_i  # (E, 3)
        dist_scalar = torch.linalg.vector_norm(disp_vec, dim=-1, keepdim=True)  # (E, 1)
        unit_vec = disp_vec / (dist_scalar + 1e-8)  # (E, 3)

        return {
            "edge_index": edge_index,
            "dist_scalar": dist_scalar,
            "unit_vec": unit_vec,
        }

    @staticmethod
    def build_rsasa_node_tensor(
        residues: list[tuple[str, str, str]], sasa_map: dict[tuple[str, str], float]
    ) -> torch.Tensor:
        """Computes normalized rSASA tensor for node features using Tien et al. (2013).

        Args:
            residues: List of node residue metadata ordered by node index in PyG graph.
            sasa_map: Dict mapping (chain_id, res_num) -> raw SASA in Å².

        Returns:
            torch.Tensor: FloatTensor of shape (num_nodes, 1) with values in [0.0, 1.0].
        """
        rsasa_values = []

        for chain_id, res_num, res_name in residues:
            raw_sasa = sasa_map.get((chain_id, str(res_num)), 0.0)
            # Fallback 200 Å² for non-standard
            max_asa = TIEN_MAX_ASA.get(res_name.upper(), DEFAULT_MAX_ASA)

            # Calculate rSASA and clamp to [0.0, 1.0]
            rsasa = min(max(raw_sasa / max_asa, 0.0), 1.0)
            rsasa_values.append([rsasa])

        return torch.tensor(rsasa_values, dtype=torch.float32)

    @staticmethod
    def _compute_inter_chain_flag(
        edge_index: torch.Tensor, nodes: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Computes binary flag indicating if an edge crosses chain boundaries.

        Args:
            edge_index: Long tensor of shape (2, E).
            nodes: List of (chain_id, res_num, res_name) node metadata of length N.

        Returns:
            torch.Tensor: Float32 tensor of shape (E, 1) where 1.0 indicates an
            inter-subunit interface edge and 0.0 indicates an intra-chain edge.
        """
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()

        # Compare chain_id (index 0 of each node metadata tuple)
        flags = [
            [1.0] if nodes[i][0] != nodes[j][0] else [0.0]
            for i, j in zip(src_indices, dst_indices)
        ]
        return torch.tensor(flags, dtype=torch.float32)

    def build_graph(
        self,
        aa_list: list[str],
        coords_np: np.ndarray,
        nodes: list[tuple[str, str, str]],
        sasa_map: dict[tuple[str, str], float],
        name: str = "",
    ) -> Data:
        """Assembles node features, topology, edge features, and coordinates into a PyG Data object.

        Args:
            aa_list: Sequence of 3-letter amino acid codes of length N.
            coords_np: NumPy array of C-alpha coordinates of shape (N, 3).
            nodes: List of (chain_id, res_num_str, res_name_3let) corresponding to PyG nodes.
            sasa_map: Dict mapping (chain_id, res_num_str) -> raw SASA in Å².
            name: Structural identifier or complex name metadata.

        Returns:
            Data: PyTorch Geometric Data instance with x (N, 23), edge_index, edge_attr, pos, and name.
        """
        # Input Sanitization Guardrail
        if not np.isfinite(coords_np).all():
            raise ValueError(
                f"Invalid coordinates in structure '{name}': input contains NaN or Inf values."
            )

        if len(aa_list) != len(coords_np):
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {len(aa_list)} amino acids but {len(coords_np)} coordinate vectors."
            )
        if len(nodes) != len(aa_list):
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {len(nodes)} residue node metadata items but {len(aa_list)} amino acids."
            )
        coords = torch.from_numpy(coords_np).float()

        # 1. Base Node Features: (N, 22) -> AA One-Hot + Sequence Scalars
        x_base = self._encode_node_feature(aa_list)

        # 2. Compute Relative SASA Node Feature: (N, 1)
        rsasa_tensor = self.build_rsasa_node_tensor(nodes, sasa_map)

        # 3. Concatenate Base Features + rSASA: (N, 22) cat (N, 1) -> (N, 23)
        x = torch.cat([x_base, rsasa_tensor], dim=1)

        # 4. Extract Geometric Vectors & Scalar Distances: (E, 3) and (E, 1)
        geom_attrs = self._compute_edge_geometric_attrs(coords_np)
        edge_index = geom_attrs["edge_index"]  # (2, E)
        unit_vec = geom_attrs["unit_vec"]  # (E, 3)
        dist_scalar = geom_attrs["dist_scalar"]  # (E, 1)

        # 5. Apply Gaussian RBF Expansion: (E, 1) -> (E, 16)
        rbf_out = self.rbf_module(dist_scalar)

        # 6. Compute Inter-Subunit Interface Flag: (E, 1)
        inter_chain_flag = self._compute_inter_chain_flag(edge_index, nodes)

        # 7. Concatenate Direction Vectors + RBF Fingerprints + Interface Flag: (E, 20)
        edge_attr = torch.cat([unit_vec, rbf_out, inter_chain_flag], dim=1)

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=coords, name=name
        )
