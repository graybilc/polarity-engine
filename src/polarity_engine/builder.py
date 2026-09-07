#!/urs/bin/env python3


import logging
import numpy as np
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
        """Converts a dense pairwise distance matrix to PyTorch Geometric COO edge_index format.

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

    def build_graph(
        self, aa_list: list[str], coords_np: np.ndarray, name: str = ""
    ) -> Data:
        """Assembles node features, topology, edge features, and coordinates into a PyG Data object.

        Args:
            aa_list: Sequence of 3-letter amino acid codes of length N.
            coords_np: NumPy array of C-alpha coordinates of shape (N, 3).
            name: Structural identifier or complex name metadata.

        Returns:
            Data: PyTorch Geometric Data instance with x, edge_index, edge_attr, pos,
            and name.
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
        coords = torch.from_numpy(coords_np).float()

        # 1. Node Features: (N, 22)
        x = self._encode_node_feature(aa_list)

        # 2. Extract Geometric Vectors & Scalar Distances: (E, 3) and (E, 1)
        geom_attrs = self._compute_edge_geometric_attrs(coords_np)
        edge_index = geom_attrs["edge_index"]  # (2, E)
        unit_vec = geom_attrs["unit_vec"]  # (E, 3)
        dist_scalar = geom_attrs["dist_scalar"]  # (E, 1)

        # 3. Apply Gaussian RBF Expansion: (E, 1) -> (E, 16)
        rbf_out = self.rbf_module(dist_scalar)

        # 4. Concatenate Direction Vectors + RBF Fingerprints: (E, 19)
        edge_attr = torch.cat([unit_vec, rbf_out], dim=1)

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=coords, name=name
        )
