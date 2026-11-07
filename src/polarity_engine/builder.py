#!/usr/bin/env python3

from typing import Dict, Any, Union
from pathlib import Path
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from polarity_engine.constants import AMINO_ACID_TO_INDEX
from polarity_engine.parsers import StructureParser

logger = logging.getLogger(__name__)


def load_resolved_site_mapping(
    mapping_json_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Loads resolved phosphorylation site mapping JSON and extracts structural attributes per site.

    Args:
        mapping_json_path (str | Path): File path to the site mapping JSON file.

    Returns:
        site_nodes (Dict[str, Dict[str, Any]]): Dictionary mapping site IDs to their residue number,
            anchor flag, and amino acid type.
    """
    path = Path(mapping_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Site mapping JSON not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    site_nodes = {}
    for site_id, site_info in data.get("sites", {}).items():
        status = site_info.get("status")
        if status == "MODELED":
            site_nodes[site_id] = {
                "res_num": site_info["target_res_num"],
                "is_anchor": False,
                "aa": site_info.get("target_aa"),
            }
        elif status == "UNMODELED":
            site_nodes[site_id] = {
                "res_num": site_info["fallback_anchor_res_num"],
                "is_anchor": True,
                "aa": site_info.get("anchor_aa"),
            }

    return site_nodes


class GaussianRBF(nn.Module):
    """
    Computes Radial Basis Function (RBF) expansions using Gaussian kernels over inter-residue distances.
    """

    def __init__(self, start: float = 0.0, stop: float = 8.0, num_gaussians: int = 16):
        """
        Initializes the Gaussian RBF module with specified distance bounds and kernel counts.

        Args:
            start (float): Minimum distance value for kernel centers. Defaults to 0.0.
            stop (float): Maximum distance value for kernel centers. Defaults to 8.0.
            num_gaussians (int): Number of Gaussian RBF channels to generate. Defaults to 16.
        """
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        delta = (stop - start) / (num_gaussians - 1)
        gamma = 1.0 / (delta**2)

        self.register_buffer("mu", mu)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Projects scalar distances into Gaussian RBF feature space.

        Args:
            dist (torch.Tensor): Tensor of pairwise Euclidean distances.

        Returns:
            torch.Tensor: Expanded Gaussian RBF tensor features.
        """
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        diff = dist - self.mu.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class ProteinGraphBuilder:
    """
    Constructs PyTorch Geometric Data graph objects from parsed protein structures
    with standardized 27-dimensional node features and 20-dimensional edge attributes.
    """

    def __init__(self, distance_cutoff: float = 8.0, num_rbf_kernels: int = 16) -> None:
        """
        Initializes the ProteinGraphBuilder with spatial cutoff and RBF expansion parameters.

        Args:
            distance_cutoff (float): Maximum Euclidean distance threshold for spatial edge creation. Defaults to 8.0.
            num_rbf_kernels (int): Number of RBF expansion channels for edge attributes. Defaults to 16.
        """
        self.distance_cutoff = distance_cutoff
        self.rbf_module = GaussianRBF(
            start=0.0, stop=distance_cutoff, num_gaussians=num_rbf_kernels
        )

    def _encode_node_feature(self, aa_list: list[str]) -> torch.Tensor:
        """
        Encodes amino acid sequence identity and normalized sequence position into base node features.

        Args:
            aa_list (list[str]): List of 3-letter amino acid residue names.

        Returns:
            torch.Tensor: Base node feature tensor combining one-hot amino acid encoding and sequence position.
        """
        aa_indices = torch.tensor(
            [
                AMINO_ACID_TO_INDEX.get(res.upper(), AMINO_ACID_TO_INDEX["UNK"])
                for res in aa_list
            ],
            dtype=torch.long,
        )
        x_one_hot = torch.nn.functional.one_hot(aa_indices, num_classes=22).float()
        seq_pos = torch.arange(len(aa_list), dtype=torch.float32) / max(
            len(aa_list) - 1, 1
        )
        seq_pos_col = seq_pos.unsqueeze(1)
        return torch.cat([x_one_hot, seq_pos_col], dim=1)

    def _distance_to_sparse_coo(
        self, dist_matrix: torch.Tensor, include_self_loops: bool = False
    ) -> tuple[torch.Tensor, int]:
        """
        Converts a pairwise distance matrix into a sparse COO format edge index tensor.

        Args:
            dist_matrix (torch.Tensor): Square matrix of pairwise Euclidean distances.
            include_self_loops (bool): Whether to include self-loop edges where distance is zero. Defaults to False.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the COO edge index tensor and total edge count.
        """
        if include_self_loops:
            mask = dist_matrix <= self.distance_cutoff
        else:
            mask = (dist_matrix <= self.distance_cutoff) & (dist_matrix > 0.0)

        edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous().long()
        num_edges = edge_index.shape[1]
        return edge_index, num_edges

    def _compute_edge_geometric_attrs(
        self, coords: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """
        Computes geometric edge attributes including distances, unit vectors, and adjacency indices.

        Args:
            coords (np.ndarray): Cartesian coordinates of C-alpha atoms of shape (N, 3).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing edge index, scalar distances, and unit direction vectors.
        """
        coords_torch = torch.from_numpy(coords).float()
        dist_matrix = torch.cdist(coords_torch, coords_torch)
        edge_index, _ = self._distance_to_sparse_coo(dist_matrix)

        pos_i = coords_torch[edge_index[0]]
        pos_j = coords_torch[edge_index[1]]

        disp_vec = pos_j - pos_i
        dist_scalar = torch.linalg.vector_norm(disp_vec, dim=-1, keepdim=True)
        unit_vec = disp_vec / (dist_scalar + 1e-8)

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
        Computes inter-chain binary flags for graph edges based on residue chain identifiers.

        Args:
            edge_index (torch.Tensor): Graph connectivity tensor of shape (2, E).
            nodes (list[tuple[str, str, str]]): List of standardized node descriptors (chain_id, res_num, res_name).

        Returns:
            torch.Tensor: Tensor of shape (E, 1) indicating whether an edge spans different chains (1.0) or the same chain (0.0).
        """
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return torch.zeros((0, 1), dtype=torch.float32)

        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()

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
        Constructs a PyTorch Geometric Data object encapsulating 27-dimensional node features
        and 20-dimensional edge attributes for a protein structure.

        Args:
            aa_list (list[str]): List of 3-letter amino acid names for each residue node.
            coords_np (np.ndarray): C-alpha Cartesian coordinates of shape (N, 3).
            b_factors_np (np.ndarray): Crystallographic temperature factors of shape (N,).
            occupancies_np (np.ndarray): Atom occupancy values of shape (N,).
            nodes (list[tuple[str, str, str]]): List of standardized node descriptors (chain_id, res_num, res_name).
            rsasa_np (np.ndarray): Relative solvent accessible surface area values of shape (N,).
            anm_msf_np (np.ndarray | None): ANM mean-square fluctuation values of shape (N,). Defaults to None.
            name (str): Identifier name for the graph object. Defaults to "".

        Returns:
            Data: PyTorch Geometric Data object containing node features `x`, edge attributes `edge_attr`,
                edge indices `edge_index`, coordinates `pos`, and structural metadata.
        """
        num_nodes = len(aa_list)

        # 1. Validate metadata and feature vector lengths
        if len(nodes) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure: len(nodes) ({len(nodes)}) != len(aa_list) ({num_nodes})"
            )
        if len(b_factors_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure: len(b_factors) ({len(b_factors_np)}) != len(aa_list) ({num_nodes})"
            )
        if len(occupancies_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure: len(occupancies) ({len(occupancies_np)}) != len(aa_list) ({num_nodes})"
            )
        if len(rsasa_np) != num_nodes:
            raise ValueError(
                f"Length mismatch in structure: len(rsasa) ({len(rsasa_np)}) != len(aa_list) ({num_nodes})"
            )

        coords_arr = np.asarray(coords_np)
        if coords_arr.shape[0] != num_nodes:
            raise ValueError(
                f"Length mismatch in structure: len(coords) ({coords_arr.shape[0]}) != len(aa_list) ({num_nodes})"
            )

        # 2. Validate numerical sanity (NaN / Inf checks)
        if np.isnan(coords_arr).any() or np.isinf(coords_arr).any():
            raise ValueError(
                "Invalid coordinates in structure: contains NaN or Inf values."
            )

        if anm_msf_np is not None:
            if len(anm_msf_np) != num_nodes:
                raise ValueError(
                    f"Length mismatch in structure: len(anm_msf_np) ({len(anm_msf_np)}) != len(aa_list) ({num_nodes})"
                )
            if np.isnan(anm_msf_np).any() or np.isinf(anm_msf_np).any():
                raise ValueError(
                    "Invalid ANM MSF in structure: contains NaN or Inf values."
                )
        else:
            anm_msf_np = np.zeros(num_nodes, dtype=np.float32)

        coords = torch.from_numpy(coords_arr).float()
        b_factors_tensor = torch.from_numpy(b_factors_np).float().unsqueeze(1)
        occupancies_tensor = torch.from_numpy(occupancies_np).float().unsqueeze(1)
        rsasa_tensor = torch.from_numpy(rsasa_np).float().unsqueeze(1)
        anm_msf_tensor = torch.from_numpy(anm_msf_np).float().unsqueeze(1)

        x_base = self._encode_node_feature(aa_list)
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

        geom_attrs = self._compute_edge_geometric_attrs(coords_arr)
        edge_index = geom_attrs["edge_index"]
        unit_vec = geom_attrs["unit_vec"]
        dist_scalar = geom_attrs["dist_scalar"]

        rbf_out = self.rbf_module(dist_scalar)
        inter_chain_flag = self._compute_inter_chain_flag(edge_index, nodes)
        edge_attr = torch.cat([unit_vec, rbf_out, inter_chain_flag], dim=1)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=coords,
            name=name,
            nodes=nodes,
        )

    def build_from_parsed_dict(
        self,
        parsed_dict: dict[str, Any],
        rsasa_np: np.ndarray | None = None,
        anm_msf_np: np.ndarray | None = None,
        site_mapping: dict | None = None,
        name: str = "",
    ) -> Data:
        """
        Builds a PyTorch Geometric protein graph directly from a pre-parsed structure dictionary.

        Args:
            parsed_dict (dict[str, Any]): Dictionary containing structural arrays and metadata from StructureParser.
            rsasa_np (np.ndarray | None): Optional pre-computed rSASA vector. Defaults to None.
            anm_msf_np (np.ndarray | None): Optional pre-computed ANM MSF vector. Defaults to None.
            site_mapping (dict | None): Optional site mapping dictionary for phosphorylation analysis. Defaults to None.
            name (str): Identifier name for the graph object. Defaults to "".

        Returns:
            Data: PyTorch Geometric Data object constructed from the parsed dictionary attributes.
        """
        if "nodes" in parsed_dict:
            nodes = parsed_dict["nodes"]
        elif "aa_residues" in parsed_dict and "all_atom_keys" in parsed_dict:
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

        if "aa_list" in parsed_dict:
            aa_list = parsed_dict["aa_list"]
        else:
            aa_list = [res_name for _, _, res_name in nodes]

        if "coords" in parsed_dict:
            coords_np = np.asarray(parsed_dict["coords"], dtype=np.float32)
        elif "ca_coords" in parsed_dict:
            coords_np = np.asarray(parsed_dict["ca_coords"], dtype=np.float32)
        else:
            raise KeyError("Neither 'coords' nor 'ca_coords' found in parsed_dict.")

        if "b_factors" in parsed_dict:
            b_factors_np = np.asarray(parsed_dict["b_factors"], dtype=np.float32)
        elif "ca_b_factors" in parsed_dict:
            b_factors_np = np.asarray(parsed_dict["ca_b_factors"], dtype=np.float32)
        else:
            b_factors_np = np.zeros(len(aa_list), dtype=np.float32)

        if "occupancies" in parsed_dict:
            occupancies_np = np.asarray(parsed_dict["occupancies"], dtype=np.float32)
        elif "ca_occupancies" in parsed_dict:
            occupancies_np = np.asarray(parsed_dict["ca_occupancies"], dtype=np.float32)
        else:
            occupancies_np = np.ones(len(aa_list), dtype=np.float32)

        if rsasa_np is None:
            if "all_atom_coords" in parsed_dict:
                try:
                    rsasa_np = StructureParser.extract_node_rsasa_vector(parsed_dict)
                except Exception as e:
                    logger.warning(
                        f"StructureParser rSASA extraction failed ({e}), falling back to zeros."
                    )
                    rsasa_np = np.zeros(len(aa_list), dtype=np.float32)
            else:
                rsasa_np = np.zeros(len(aa_list), dtype=np.float32)

        rsasa_np = np.clip(rsasa_np, 0.0, 1.0)

        if anm_msf_np is None:
            if "anm_msf" in parsed_dict:
                anm_msf_np = np.asarray(parsed_dict["anm_msf"], dtype=np.float32)
            else:
                anm_msf_np = np.zeros(len(aa_list), dtype=np.float32)

        if site_mapping is None and "site_mapping" in parsed_dict:
            site_mapping = parsed_dict["site_mapping"]

        graph_data = self.build_graph(
            aa_list=aa_list,
            coords_np=coords_np,
            b_factors_np=b_factors_np,
            occupancies_np=occupancies_np,
            nodes=nodes,
            rsasa_np=rsasa_np,
            anm_msf_np=anm_msf_np,
            name=name,
        )

        if site_mapping:
            graph_data.site_mapping = site_mapping

        return graph_data
