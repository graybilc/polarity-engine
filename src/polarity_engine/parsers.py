#!/usr/bin/env python3


import freesasa
import logging
import numpy as np
import scipy.linalg as la
import torch

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from pathlib import Path
from typing import Any, Dict, Union

from polarity_engine.constants import (
    DEFAULT_MAX_ASA,
    DEFAULT_VDW_RADIUS,
    ELEMENT_RADII,
    MAX_SASA_TIEN,
)

# Configure log
logger = logging.getLogger(__name__)


class FastaParser:
    """
    Parses and validates FASTA sequence files from scratch.
    """

    STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
    NON_STANDARD_AA = "UOBZXJ*"

    # Convert to a set for efficient O(1) membership lookups
    VALID_AA = set(STANDARD_AA + NON_STANDARD_AA)
    NON_STANDARD_AA_SET = set(NON_STANDARD_AA)

    @classmethod
    def parse_fasta(cls, file_path: str | Path) -> dict[str, str]:
        """
        Parses a FASTA file line-by-line from scratch.

        Args:
            file_path (str | Path): File path to the desired FASTA file.
        Returns:
            sequences (dict[str, str]): A dictionary mapping header strings to clean sequence strings.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            msg = f"Target FASTA file not found or invalid: '{file_path.resolve()}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        sequences = {}
        current_header = None
        current_sequence = []

        with open(file_path, "r", encoding="utf-8") as fasta_reader:
            for line in fasta_reader:
                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):
                    if current_header:
                        raw_sequence = "".join(current_sequence)
                        sequences[current_header] = cls._validate_amino_acid_sequence(
                            raw_sequence, current_header
                        )
                    current_header = line[1:].strip()
                    current_sequence = []
                else:
                    if current_header is None:
                        msg = f"Malformed FASTA in '{file_path.name}': Found sequence data before a header."
                        logger.error(msg)
                        raise ValueError(msg)
                    current_sequence.append(line.upper())

            if current_header:
                raw_sequence = "".join(current_sequence)
                sequences[current_header] = cls._validate_amino_acid_sequence(
                    raw_sequence, current_header
                )

        # --- Post-Parsing Validation ---
        if not sequences:
            msg = f"Malformed FASTA: File '{file_path.name}' contains no entries."
            logger.error(msg)
            raise ValueError(msg)

        return sequences

    @classmethod
    def _validate_amino_acid_sequence(cls, sequence: str, header: str) -> str:
        """
        Validate all the amino acids in the given sequence are valid.
        Non-standard amino acids are recorded in the log framework.

        Args:
            sequence (str): Amino acid sequence to be validated.
            header (str): The corresponding fasta header for explicit error tracking.
        Returns:
            cleaned_seq (str): Validated amino acid sequence.
        """
        cleaned_seq = "".join(sequence.split()).upper()

        if not cleaned_seq:
            msg = f"Malformed FASTA: Empty sequence string encountered under header '{header}'."
            logger.error(msg)
            raise ValueError(msg)

        for i, aa in enumerate(cleaned_seq):
            if aa not in cls.VALID_AA:
                msg = f"Invalid amino acid '{aa}' found at position {i + 1} under header '{header}'."
                logger.error(msg)
                raise ValueError(msg)

            if aa in cls.NON_STANDARD_AA_SET:
                logger.warning(
                    f"Non-standard amino acid '{aa}' found at position {i + 1} under header '{header}'."
                )

        return cleaned_seq


class StructureParser:
    """
    Parses mmCIF protein structure files to extract 3D coordinates
    and biophysical surface properties (SASA).

    Handles raw file I/O, coordinate extractions, and native C-binding
    surface accessibility calculations via FreeSASA using a high-speed
    MMCIF2Dict array-extraction fast path.
    """

    @classmethod
    def _validate_structure_file(cls, file_path: str | Path) -> str:
        """
        Verify the provided file exists and has a valid mmCIF extension.

        Args:
            file_path (str | Path): File path to the desired structure file.
        Returns:
            file_extension (str): Lowercased file extension.
        """
        file_path = Path(file_path)

        if not file_path.is_file():
            msg = f"Target structure file not found: '{file_path.resolve()}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        file_ext = file_path.suffix.lower()
        if file_ext not in [".cif", ".mmcif"]:
            msg = f"Unsupported file format found: '{file_ext}'. Expected mmCIF (.cif / .mmcif)."
            logger.error(msg)
            raise ValueError(msg)

        return file_ext

    @classmethod
    def _load_and_inspect(cls, file_path: Path) -> tuple[list[str], dict]:
        """
        Validates the structure file, loads it, and extracts unique chain IDs
        alongside the raw MMCIF2Dict object.

        Args:
            file_path (str | Path): File path to the desired structure file.
        Returns:
            tuple[list[str], dict]: A list of unique chain IDs and the loaded MMCIF2Dict.
        """
        # Validate file existence and extension first
        cls._validate_structure_file(file_path)
        path_obj = Path(file_path)

        try:
            mmcif_dict = MMCIF2Dict(str(path_obj))
            key = None
            if "_atom_site.auth_asym_id" in mmcif_dict:
                key = "_atom_site.auth_asym_id"
            elif "_atom_site.label_asym_id" in mmcif_dict:
                key = "_atom_site.label_asym_id"
            else:
                raise ValueError(
                    "Missing essential chain identifier: neither '_atom_site.auth_asym_id' "
                    "nor '_atom_site.label_asym_id' found in mmCIF dictionary."
                )

            seen = set()
            chains = [c for c in mmcif_dict[key] if not (c in seen or seen.add(c))]
            return chains, mmcif_dict
        except Exception as e:
            raise ValueError(f"Malformed mmCIF file in '{path_obj.name}': {e}")

    @classmethod
    def get_all_atom_coordinates(
        cls, file_path: str | Path, chain_ids: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Main entry point to extract C-alpha backbone and heavy-atom coordinates per chain.

        Args:
            file_path: File path to the target mmCIF structure file.
            chain_ids: List of chain identifiers to parse. If None, parses all available chains.

        Returns:
            dict[str, dict[str, Any]]: Dictionary mapping chain_id to its parsed coordinate dictionaries.
        """
        path_obj = Path(file_path)

        available_chains, loaded_data = cls._load_and_inspect(path_obj)
        if not available_chains:
            raise ValueError(f"No chains found in structural file '{path_obj.name}'")

        target_chains = (
            available_chains
            if chain_ids is None
            else [c for c in chain_ids if c in available_chains]
        )

        if chain_ids is not None and not target_chains:
            raise ValueError(
                f"None of requested chain_ids {chain_ids} were found in"
                f" '{path_obj.name}'. Available chains: {available_chains}"
            )

        results = {}
        for c_id in target_chains:
            results[c_id] = cls._parse_mmcif_fast_path(loaded_data, c_id)

        return results

    @classmethod
    def _parse_mmcif_fast_path(cls, mmcif_dict: dict, chain_id: str) -> dict[str, Any]:
        """
        Parses mmCIF coordinates from a pre-loaded MMCIF2Dict.

        Extracts all heavy atoms for all-atom SASA calculations alongside
        C-alpha backbone node attributes.

        Args:
            mmcif_dict (dict): MMCIF2Dict object for the target project.
            chain_id (str): Name used for the distinct macromolecule in the structure file.

        Returns:
            dict[str, Any]: Dict containing:
                - 'ca_coords': (N, 3) float32 array of C-alpha coordinates.
                - 'ca_b_factors': (N,) float32 array of C-alpha B-factors.
                - 'ca_occupancies': (N,) float32 array of C-alpha occupancies.
                - 'aa_residues': List of (seq_num_str, res_name) tuples for
                    C-alpha nodes.
                - 'all_atom_coords': (M, 3) float64 array of all heavy atom
                    coordinates.
                - 'all_atom_keys': List of (chain_id, seq_num_str) tuples of
                    length M.
                - 'all_atom_names': List of atom names (e.g., 'CA', 'N', 'CB')
                    of length M.
                - 'all_atom_res_names': List of 3-letter residue names of length
                    M.
        """
        seq_key = (
            "_atom_site.auth_seq_id"
            if "_atom_site.auth_seq_id" in mmcif_dict
            else "_atom_site.label_seq_id"
        )

        comp_id_col = "_atom_site.auth_comp_id"
        if comp_id_col not in mmcif_dict:
            comp_id_col = "_atom_site.label_comp_id"

        if comp_id_col not in mmcif_dict:
            raise ValueError(
                "Malformed mmCIF structure: Missing residue identifier column."
            )

        required_keys = [
            "_atom_site.group_PDB",
            "_atom_site.auth_asym_id",
            "_atom_site.label_atom_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            comp_id_col,
            seq_key,
            "_atom_site.B_iso_or_equiv",
            "_atom_site.occupancy",
        ]

        for key in required_keys:
            if key not in mmcif_dict:
                msg = f"Malformed mmCIF structure: Missing required data field '{key}'"
                logger.error(msg)
                raise ValueError(msg)

        # Normalize single-element string values into lists for safe zipping
        cols = {
            k: [mmcif_dict[k]] if isinstance(mmcif_dict[k], str) else mmcif_dict[k]
            for k in required_keys
        }

        ca_coordinates, aa_residues, ca_b_factors, ca_occupancies = [], [], [], []
        all_atom_coords, all_atom_keys, all_atom_names, all_atom_res_names = (
            [],
            [],
            [],
            [],
        )

        for group, chain, atom_name, x, y, z, aa, seq_num, b_val, occ_val in zip(
            cols["_atom_site.group_PDB"],
            cols["_atom_site.auth_asym_id"],
            cols["_atom_site.label_atom_id"],
            cols["_atom_site.Cartn_x"],
            cols["_atom_site.Cartn_y"],
            cols["_atom_site.Cartn_z"],
            cols[comp_id_col],
            cols[seq_key],
            cols["_atom_site.B_iso_or_equiv"],
            cols["_atom_site.occupancy"],
        ):
            clean_aa = str(aa).strip().upper()
            is_valid_group = (group == "ATOM") or (
                group == "HETATM" and clean_aa in {"SEP", "TPO", "PTR"}
            )

            if not (is_valid_group and chain == chain_id):
                continue

            clean_atom_name = str(atom_name).strip()

            if clean_atom_name.startswith(("H", "D", "1H", "2H", "3H")):
                continue

            try:
                coord = [float(x), float(y), float(z)]
                res_num_str = str(seq_num)
                res_name_str = clean_aa

                all_atom_coords.append(coord)
                all_atom_keys.append((chain_id, res_num_str))
                all_atom_names.append(clean_atom_name)
                all_atom_res_names.append(res_name_str)

                if clean_atom_name == "CA":
                    ca_coordinates.append(coord)
                    aa_residues.append((res_num_str, res_name_str))
                    ca_b_factors.append(float(b_val))
                    ca_occupancies.append(float(occ_val))

            except ValueError as e:
                msg = f"Non-numeric spatial coordinates encountered in mmCIF dictionary: {e}"
                logger.error(msg)
                raise ValueError(msg) from e

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return {
            "ca_coords": np.array(ca_coordinates, dtype=np.float32),
            "aa_residues": aa_residues,
            "ca_b_factors": np.array(ca_b_factors, dtype=np.float32),
            "ca_occupancies": np.array(ca_occupancies, dtype=np.float32),
            "all_atom_coords": np.array(all_atom_coords, dtype=np.float64),
            "all_atom_keys": all_atom_keys,
            "all_atom_names": all_atom_names,
            "all_atom_res_names": all_atom_res_names,
        }

    @classmethod
    def save_parsed_dict(
        cls, parsed_dict: Dict[str, Any], output_path: Union[str, Path]
    ) -> Path:
        """
        Serializes and saves the pre-parsed dictionary to disk using PyTorch serialization.

        Args:
            parsed_dict (Dict[str, Any]): Dictionary returned by parsing methods.
            output_path (Union[str, Path]): Destination file path.

        Returns:
            Path: Path object pointing to the saved file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(parsed_dict, out_path)
        return out_path

    @classmethod
    def load_parsed_dict(cls, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads a pre-parsed dictionary from disk.

        Args:
            input_path (Union[str, Path]): File path to the serialized PyTorch dictionary (.pt).

        Returns:
            Dict[str, Any]: Loaded dictionary containing atom coordinates and residue metadata.
        """
        in_path = Path(input_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Parsed dictionary file not found at: {in_path}")

        parsed_dict = torch.load(in_path, weights_only=False)
        return parsed_dict

    @classmethod
    def _assign_vdw_radii(
        cls, atom_names: list[str], res_names: list[str]
    ) -> list[float]:
        """
        Assigns atomic Van der Waals radii (Å) based on heavy-atom element types.

        Maps extracted atom names and residue types to empirical physical radii
        defined in `constants.ELEMENT_RADII`. Handles structural metal species
        (e.g., Fe, Zn, Mg) and defaults to `constants.DEFAULT_VDW_RADIUS` for
        unmapped heavy elements.

        Args:
            atom_names (list[str]): List of length M containing atom identifiers
                (e.g., 'CA', 'N', 'CB', 'FE').
            res_names (list[str]): List of length M containing 3-letter residue
                names (e.g., 'ALA', 'TRP', 'HEM').

        Returns:
            list[float]: List of length M containing atomic Van der Waals
                radii in Angstroms (Å) for FreeSASA Option A calculations.
        """
        radii = []
        for atom_name, res_name in zip(atom_names, res_names):
            clean_atom = atom_name.upper().strip()
            if clean_atom.startswith("FE"):
                elem = "FE"
            elif clean_atom.startswith("ZN"):
                elem = "ZN"
            elif clean_atom.startswith("MG"):
                elem = "MG"
            else:
                elem = clean_atom[0]

            radii.append(ELEMENT_RADII.get(elem, DEFAULT_VDW_RADIUS))
        return radii

    @classmethod
    def _compute_allatom_sasa_dict(
        cls, parsed_data: dict[str, Any]
    ) -> dict[tuple[str, str], float]:
        """
        Computes All-Atom SASA in-memory using freesasa.calcCoord and aggregates
        atomic surface area values per residue.

        Args:
            parsed_data (dict[str, Any]): Dictionary returned by parsing methods.

        Returns:
            dict[tuple[str, str], float]: Mapping of (chain_id, res_number_str) -> raw SASA (Å²).
        """
        all_atom_coords: np.ndarray = parsed_data["all_atom_coords"]
        all_atom_keys: list[tuple[str, str]] = parsed_data["all_atom_keys"]
        all_atom_names: list[str] = parsed_data["all_atom_names"]
        all_atom_res_names: list[str] = parsed_data["all_atom_res_names"]

        if all_atom_coords.size == 0:
            raise ValueError("Empty coordinate array passed to FreeSASA calculation.")

        radii = cls._assign_vdw_radii(all_atom_names, all_atom_res_names)
        flat_coords = all_atom_coords.flatten().tolist()

        try:
            result = freesasa.calcCoord(flat_coords, radii)
        except Exception as e:
            raise ValueError(
                f"freesasa.calcCoord failed during C execution: {e}"
            ) from e

        residue_sasa_map: dict[tuple[str, str], float] = {}
        for i in range(result.nAtoms()):
            key = all_atom_keys[i]
            atom_area = result.atomArea(i)
            residue_sasa_map[key] = residue_sasa_map.get(key, 0.0) + atom_area

        return residue_sasa_map

    @staticmethod
    def _compute_rsasa_vector(
        nodes: list[tuple[str, str, str]], sasa_map: dict[tuple[str, str], float]
    ) -> np.ndarray:
        """
        Normalizes pre-computed raw SASA values into a C-alpha-aligned rSASA vector.

        Args:
            nodes: Length $N$ list of standardized residue node descriptors.
            sasa_map: Mapping of `(chain_id, res_num)` to absolute SASA values.

        Returns:
            np.ndarray: Float32 array of shape `(N,)` containing normalized relative SASA values.
        """
        rsasa_list = []
        for c_id, res_num, res_name in nodes:
            key = (c_id, res_num)
            raw_sasa = sasa_map.get(key, 0.0)
            max_sasa = MAX_SASA_TIEN.get(res_name.upper(), DEFAULT_MAX_ASA)
            normalized_rsasa = min(max(raw_sasa / max_sasa, 0.0), 1.0)
            rsasa_list.append(normalized_rsasa)

        return np.array(rsasa_list, dtype=np.float32)

    @classmethod
    def extract_node_rsasa_vector(
        cls,
        parsed_data: dict[str, Any],
    ) -> np.ndarray:
        """
        Computes per-residue normalized rSASA values (rSASA in [0, 1]) aligned exactly
        with the downstream C-alpha node order ('ca_coords').

        Args:
            parsed_data (dict[str, Any]): Output dictionary from parsing methods.

        Returns:
            np.ndarray: (N,) float32 array of relative SASA values aligned with C-alpha nodes.
        """
        raw_sasa_map = cls._compute_allatom_sasa_dict(parsed_data)
        aa_residues: list[tuple[str, str]] = parsed_data["aa_residues"]
        chain_id = parsed_data["all_atom_keys"][0][0]

        nodes = [(chain_id, res_num, res_name) for res_num, res_name in aa_residues]

        return cls._compute_rsasa_vector(nodes, raw_sasa_map)

    @staticmethod
    def _aggregate_chains(
        chain_parsed_data: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """
        Aggregates multi-chain structural dictionaries into unified complex-level buffers.

        Args:
            chain_parsed_data: Dictionary mapping chain identifiers to parsed feature dictionaries.

        Returns:
            A tuple `(nodes_data, heavy_data)` containing concatenated complex-wide structures:
                - `nodes_data` (dict[str, list]): C-alpha node buffers of length N:
                    - "aa_list" (list[str]): 3-letter residue names.
                    - "coords" (list[np.ndarray]): 3D C-alpha coordinate vectors.
                    - "b_factors" (list[float]): C-alpha temperature factors.
                    - "occupancies" (list[float]): C-alpha atom occupancies.
                    - "nodes" (list[tuple[str, str, str]]): Standardized node descriptors
                        `(chain_id, res_num, res_name)`.
                - `heavy_data` (dict[str, Any]): Complex-wide heavy-atom payload for FreeSASA:
                    - "all_atom_coords" (np.ndarray): Shape (M, 3) float64 matrix of all heavy-atom
                        Cartesian coordinates across all aggregated chains.
                    - "all_atom_keys" (list[tuple[str, str]]): Length M list of `(chain_id, res_num)`
                        tuples for atomic-to-residue SASA aggregation.
                    - "all_atom_names" (list[str]): Length M list of cleaned atom identifiers
                        (e.g., "CA", "N", "CB").
                    - "all_atom_res_names" (list[str]): Length M list of 3-letter residue names.
        """
        all_aa, all_coords, all_b_factors, all_occupancies, all_nodes = (
            [],
            [],
            [],
            [],
            [],
        )
        all_heavy_coords, all_heavy_keys, all_heavy_names, all_heavy_res_names = (
            [],
            [],
            [],
            [],
        )

        for c_id, chain_data in chain_parsed_data.items():
            coords_arr = chain_data["ca_coords"]
            residues = chain_data["aa_residues"]
            b_factors = chain_data["ca_b_factors"]
            occupancies = chain_data["ca_occupancies"]

            for i in range(len(coords_arr)):
                res_item = residues[i]
                if hasattr(res_item, "get_resname"):
                    res_name = res_item.get_resname().strip()
                    res_num = str(res_item.id[1])
                elif isinstance(res_item, (tuple, list)):
                    res_num = str(res_item[0])
                    res_name = str(res_item[1])
                else:
                    res_name = str(res_item)
                    res_num = str(i + 1)

                all_coords.append(coords_arr[i])
                all_aa.append(res_name)
                all_b_factors.append(b_factors[i])
                all_occupancies.append(occupancies[i])
                all_nodes.append((c_id, res_num, res_name))

            all_heavy_coords.append(chain_data["all_atom_coords"])
            all_heavy_keys.extend(chain_data["all_atom_keys"])
            all_heavy_names.extend(chain_data["all_atom_names"])
            all_heavy_res_names.extend(chain_data["all_atom_res_names"])

        nodes_data = {
            "aa_list": all_aa,
            "coords": all_coords,
            "b_factors": all_b_factors,
            "occupancies": all_occupancies,
            "nodes": all_nodes,
        }
        heavy_data = {
            "all_atom_coords": np.vstack(all_heavy_coords),
            "all_atom_keys": all_heavy_keys,
            "all_atom_names": all_heavy_names,
            "all_atom_res_names": all_heavy_res_names,
        }

        return nodes_data, heavy_data

    @classmethod
    def _compute_anm_msf(
        cls,
        ca_coords: np.ndarray | list[np.ndarray],
        cutoff: float = 15.0,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """
        Computes residue-level Mean-Square Fluctuations (MSF) using a C-alpha ANM.

        Args:
            ca_coords: Array or list of shape (N, 3) containing C-alpha Cartesian coordinates.
            cutoff: Distance threshold in Angstroms for inter-residue spring connections.
            gamma: Uniform spring constant.

        Returns:
            Normalized MSF array of shape (N,) scaled to [0, 1].
        """
        coords_arr = np.asarray(ca_coords, dtype=np.float64)
        N = len(coords_arr)
        if N < 3:
            return np.zeros(N, dtype=np.float32)

        diffs = coords_arr[:, None, :] - coords_arr[None, :, :]  # Shape: (N, N, 3)
        dists = np.linalg.norm(diffs, axis=-1)  # Shape: (N, N)

        # Assemble 3N x 3N Hessian Matrix
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i != j and dists[i, j] <= cutoff:
                    r_ij = diffs[i, j]
                    d_ij = dists[i, j]
                    K_ij = -(gamma / (d_ij**2)) * np.outer(r_ij, r_ij)
                    H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = K_ij
                    H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= K_ij

        eigenvalues, eigenvectors = la.eigh(H)
        valid_mask = eigenvalues > 1e-5
        if not np.any(valid_mask):
            return np.zeros(N, dtype=np.float32)

        slow_evals = eigenvalues[valid_mask]  # Shape: (num_valid_modes,)
        # Shape: (3N, num_valid_modes)
        slow_evecs = eigenvectors[:, valid_mask]

        # Shape: (N, 3, num_valid_modes)
        evecs_3d = slow_evecs.reshape(N, 3, -1)
        sq_disp = np.sum(evecs_3d**2, axis=1)  # Shape: (N, num_valid_modes)
        msf = np.sum(sq_disp / slow_evals, axis=1)

        min_val, max_val = np.min(msf), np.max(msf)
        if max_val - min_val > 1e-8:
            msf_normalized = (msf - min_val) / (max_val - min_val)
        else:
            msf_normalized = np.zeros_like(msf)

        return msf_normalized.astype(np.float32)

    @staticmethod
    def _validate_parsed_output(parsed_output: dict[str, Any], filename: str) -> None:
        """
        Validates that output feature vectors satisfy strict element alignment contracts.

        Args:
            parsed_output: The final dictionary assembled by `parse`.
            filename: Source structure filename.
        """
        n_nodes = len(parsed_output["aa_list"])
        if not (
            len(parsed_output["coords"])
            == len(parsed_output["nodes"])
            == len(parsed_output["b_factors"])
            == len(parsed_output["occupancies"])
            == len(parsed_output["rsasa"])
            == len(parsed_output["anm_msf"])
            == n_nodes
        ):
            raise ValueError(
                f"Internal length mismatch in residue node features for '{filename}'."
            )

        m_heavy = len(parsed_output["all_atom_coords"])
        if not (
            len(parsed_output["all_atom_keys"])
            == len(parsed_output["all_atom_names"])
            == len(parsed_output["all_atom_res_names"])
            == m_heavy
        ):
            raise ValueError(
                f"Internal length mismatch in heavy-atom arrays for '{filename}'."
            )

    @classmethod
    def parse(
        cls,
        file_path: str | Path,
        chain_ids: list[str] | None = None,
        ref_cif_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Parses mmCIF structural files into unified node feature arrays
        and complex-wide biophysical metrics in a single in-memory pass.

        Args:
            file_path: Absolute or relative path to target mmCIF file.
            chain_ids: Optional list of specific chain identifiers to retain.
            ref_cif_path: Optional path to reference template structure.
        """
        path_obj = Path(file_path)
        chain_parsed_data = cls.get_all_atom_coordinates(path_obj, chain_ids=chain_ids)

        nodes_data, heavy_data = cls._aggregate_chains(chain_parsed_data)

        if ref_cif_path and Path(ref_cif_path).resolve() != path_obj.resolve():
            logger.info(
                f"Using reference structure '{ref_cif_path}' for coordinate reference."
            )

        sasa_map = cls._compute_allatom_sasa_dict(heavy_data)
        rsasa_vec = cls._compute_rsasa_vector(nodes_data["nodes"], sasa_map)
        ca_coords = nodes_data["coords"]
        msf_features = cls._compute_anm_msf(ca_coords)

        parsed_output = {
            "aa_list": nodes_data["aa_list"],
            "coords": np.array(nodes_data["coords"], dtype=np.float32),
            "b_factors": np.array(nodes_data["b_factors"], dtype=np.float32),
            "occupancies": np.array(nodes_data["occupancies"], dtype=np.float32),
            "nodes": nodes_data["nodes"],
            "sasa_map": sasa_map,
            "rsasa": rsasa_vec,
            "anm_msf": msf_features,
            "all_atom_coords": heavy_data["all_atom_coords"],
            "all_atom_keys": heavy_data["all_atom_keys"],
            "all_atom_names": heavy_data["all_atom_names"],
            "all_atom_res_names": heavy_data["all_atom_res_names"],
        }

        cls._validate_parsed_output(parsed_output, path_obj.name)
        return parsed_output
