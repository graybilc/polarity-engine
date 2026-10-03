#!/usr/bin/env python3

import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Any

from polarity_engine.constants import AMINO_ACID_TO_INDEX

# Configure logger
logger = logging.getLogger(__name__)


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
    """
    End-to-end pipeline for constructing PyTorch Geometric Data objects from parsed structure dicts.

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
        """
        Computes sparse COO topology, displacement vectors, scalar distances, and unit direction vectors.

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
    def _compute_inter_chain_flag(
        edge_index: torch.Tensor, nodes: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """
        Computes binary flag indicating if an edge crosses chain boundaries.

        Args:
            edge_index: Long tensor of shape (2, E).
            nodes: List of (chain_id, res_num, res_name) node metadata of length N.

        Returns:
            torch.Tensor: Float32 tensor of shape (E, 1) where 1.0 indicates an
            inter-subunit interface edge and 0.0 indicates an intra-chain edge.
        """
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return torch.zeros((0, 1), dtype=torch.float32)

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
        b_factors_np: np.ndarray,
        occupancies_np: np.ndarray,
        nodes: list[tuple[str, str, str]],
        rsasa_np: np.ndarray,
        anm_msf_np: np.ndarray | None = None,
        name: str = "",
    ) -> Data:
        """
        Assembles node features, topology, edge features, and coordinates into a PyG Data object.

        Args:
            aa_list: Sequence of 3-letter amino acid codes of length N.
            coords_np: NumPy array of C-alpha coordinates of shape (N, 3).
            b_factors_np: NumPy array of residue B-factors of shape (N,).
            occupancies_np: NumPy array of residue occupancies of shape (N,).
            nodes: List of (chain_id, res_num_str, res_name_3let) corresponding to PyG nodes.
            rsasa_np: NumPy array of relative SASA values of shape (N,).
            anm_msf_np: NumPy array of normalized Mean-Square Fluctuations of shape (N,).
                If None, defaults to zeros of shape (N,).
            name: Structural identifier or complex name metadata.

        Returns:
            Data: PyTorch Geometric Data instance with x (N, 26), edge_index, edge_attr, pos, and name.
        """
        num_nodes = len(aa_list)

        if anm_msf_np is None:
            anm_msf_np = np.zeros(num_nodes, dtype=np.float32)

        # 1. Input Sanitization Guardrails
        if not np.isfinite(coords_np).all():
            raise ValueError(
                f"Invalid coordinates in structure '{name}': input contains NaN or Inf values."
            )
        if not np.isfinite(b_factors_np).all():
            raise ValueError(
                f"Invalid B-factors in structure '{name}': input contains NaN or Inf values."
            )
        if not np.isfinite(occupancies_np).all():
            raise ValueError(
                f"Invalid occupancies in structure '{name}': input contains NaN or Inf values."
            )
        if not np.isfinite(rsasa_np).all():
            raise ValueError(
                f"Invalid rSASA in structure '{name}': input contains NaN or Inf values."
            )
        if not np.isfinite(anm_msf_np).all():
            raise ValueError(
                f"Invalid ANM MSF in structure '{name}': input contains NaN or Inf values."
            )

        if len(coords_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {num_nodes} amino acids but {len(coords_np)} coordinate vectors."
            )
        if len(b_factors_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {num_nodes} amino acids but {len(b_factors_np)} B-factor values."
            )
        if len(occupancies_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {num_nodes} amino acids but {len(occupancies_np)} occupancy values."
            )
        if len(rsasa_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {num_nodes} amino acids but {len(rsasa_np)} rSASA values."
            )
        if len(anm_msf_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {num_nodes} amino acids but {len(anm_msf_np)} ANM MSF values."
            )
        if len(nodes) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure '{name}': "
                f"got {len(nodes)} residue node metadata items but {num_nodes} amino acids."
            )

        coords = torch.from_numpy(coords_np).float()
        b_factors_tensor = torch.from_numpy(b_factors_np).float().unsqueeze(1)  # (N, 1)
        occupancies_tensor = (
            torch.from_numpy(occupancies_np).float().unsqueeze(1)
        )  # (N, 1)
        rsasa_tensor = torch.from_numpy(rsasa_np).float().unsqueeze(1)  # (N, 1)
        anm_msf_tensor = torch.from_numpy(anm_msf_np).float().unsqueeze(1)  # (N, 1)

        # 2. Base Node Features: (N, 22) -> AA One-Hot + Sequence Scalars
        x_base = self._encode_node_feature(aa_list)

        # 3. Concatenate Features: (N, 22) + (N, 1) + (N, 1) + (N, 1) + (N, 1) -> (N, 26)
        x = torch.cat(
            [
                x_base,
                rsasa_tensor,
                b_factors_tensor,
                occupancies_tensor,
                anm_msf_tensor,
            ],
            dim=1,
        )

        # 4. Extract Geometric Vectors & Scalar Distances: (E, 3) and (E, 1)
        geom_attrs = self._compute_edge_geometric_attrs(coords_np)
        edge_index = geom_attrs["edge_index"]  # (2, E)
        unit_vec = geom_attrs["unit_vec"]  # (E, 3)
        dist_scalar = geom_attrs["dist_scalar"]  # (E, 1)

        # 5. Apply Gaussian RBF Expansion: (E, 1) -> (E, 16)
        rbf_out = self.rbf_module(dist_scalar)

        # 6. Compute Inter-Subunit Interface Flag: (E, 1)
        inter_chain_flag = self._compute_inter_chain_flag(edge_index, nodes)

        # 7. Concatenate Edge Features: (E, 20)
        edge_attr = torch.cat([unit_vec, rbf_out, inter_chain_flag], dim=1)

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=coords, name=name
        )

    def build_from_parsed_dict(
        self,
        parsed_dict: dict[str, Any],
        rsasa_np: np.ndarray,
        anm_msf_np: np.ndarray | None = None,
        name: str = "",
    ) -> Data:
        """
        Unpacks a parsed or mutated structure dictionary and delegates to build_graph.

        Supports both multi-chain unified outputs (from `StructureParser.parse`) and
        single-chain legacy outputs.

        Args:
            parsed_dict: Dictionary returned by `StructureParser.parse` or
                `mutate_structure_dict` containing node feature vectors and heavy-atom payloads.
            rsasa_np: Relative SASA values array of shape (N,).
            anm_msf_np: ANM Mean-Square Fluctuation values array of shape (N,).
                If None, defaults to zero or fallback values inside build_graph.
            name: Identifier for the structure (e.g., 'state_101').

        Returns:
            Data: PyTorch Geometric graph data object.
        """
        # 1. Extract nodes metadata (chain_id, res_num, res_name)
        if "nodes" in parsed_dict:
            nodes = parsed_dict["nodes"]
        elif "aa_residues" in parsed_dict and "all_atom_keys" in parsed_dict:
            # Fallback for single-chain legacy dicts: pull chain_id from first heavy atom
            chain_id = (
                parsed_dict["all_atom_keys"][0][0]
                if parsed_dict["all_atom_keys"]
                else "A"
            )
            nodes = [
                (chain_id, seq_num, res_name)
                for seq_num, res_name in parsed_dict["aa_residues"]
            ]
        else:
            raise KeyError(
                "Neither 'nodes' nor valid ('aa_residues', 'all_atom_keys') found in parsed_dict."
            )

        # 2. Extract amino acid sequence list (length N)
        if "aa_list" in parsed_dict:
            aa_list = parsed_dict["aa_list"]
        else:
            aa_list = [res_name for _, _, res_name in nodes]

        # 3. Extract C-alpha backbone coordinates (length N)
        if "coords" in parsed_dict:
            coords_np = np.asarray(parsed_dict["coords"], dtype=np.float32)
        elif "ca_coords" in parsed_dict:
            coords_np = np.asarray(parsed_dict["ca_coords"], dtype=np.float32)
        else:
            raise KeyError("Neither 'coords' nor 'ca_coords' found in parsed_dict.")

        # 4. Extract B-factors (length N)
        if "b_factors" in parsed_dict:
            b_factors_np = np.asarray(parsed_dict["b_factors"], dtype=np.float32)
        elif "ca_b_factors" in parsed_dict:
            b_factors_np = np.asarray(parsed_dict["ca_b_factors"], dtype=np.float32)
        else:
            b_factors_np = np.zeros(len(aa_list), dtype=np.float32)

        # 5. Extract occupancies (length N)
        if "occupancies" in parsed_dict:
            occupancies_np = np.asarray(parsed_dict["occupancies"], dtype=np.float32)
        elif "ca_occupancies" in parsed_dict:
            occupancies_np = np.asarray(parsed_dict["ca_occupancies"], dtype=np.float32)
        else:
            occupancies_np = np.ones(len(aa_list), dtype=np.float32)

        # 6. Delegate directly to build_graph
        return self.build_graph(
            aa_list=aa_list,
            coords_np=coords_np,
            b_factors_np=b_factors_np,
            occupancies_np=occupancies_np,
            nodes=nodes,
            rsasa_np=rsasa_np,
            anm_msf_np=anm_msf_np,
            name=name,
        )
