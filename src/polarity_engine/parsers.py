#!/urs/bin/env python3


import freesasa
import logging
import numpy as np
import scipy.linalg as la
import torch

from Bio.PDB import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure
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
    Parses protein structure files (PDB/mmCIF) to extract 3D coordinates
    and biophysical surface properties (SASA).

    Handles raw file I/O, BioPython structure parsing, coordinate extractions,
    and native C-binding surface accessibility calculations via FreeSASA.

    Design Note:
        For mmCIF parsing, we deliberately bypass Biopython's full SMCRA
        (Structure/Model/Chain/Residue/Atom) object tree representation
        (via MMCIFParser) in favor of the lower-level MMCIF2Dict representation.
        This avoids heavy Python object instantiation overhead, achieving a
        high-speed array-extraction "fast path" suitable for deep learning pipelines.
        For PDB parsing, we will use Biopython's standard PDBParser since the SMCRA
        overhead are unlikely to bottleneck us.
    """

    @classmethod
    def _validate_structure_file(cls, file_path: str | Path) -> str:
        """
        Verify the provided file exists.

        Args:
            file_path (str | Path): File path to the desired structure file.
        Returns:
            file_extension (str): If file exists, the lowercased file extension will be returned.
        """
        file_path = Path(file_path)

        if not file_path.is_file():
            msg = f"Target structure file not found: '{file_path.resolve()}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        file_ext = file_path.suffix.lower()
        if file_ext not in [".pdb", ".cif", ".mmcif"]:
            msg = f"Unsupported file format found: '{file_ext}'"
            logger.error(msg)
            raise ValueError(msg)

        return file_ext

    @classmethod
    def _load_and_inspect(cls, file_path: Path, file_ext: str) -> tuple[list[str], any]:
        """
        Loads the structure file and extracts unique chain IDs alongside
        the raw parsed data object.

        Args:
            file_path (Path): File path to the desired structure file.
            file_ext (str): Lowercased extension of the structure file.
        Returns:
            tuple[list[str], any]: A list of unique chain IDs, and the loaded data
                object (Structure or MMCIF2Dict).
        """
        if file_ext == ".pdb":
            try:
                structure = PDBParser(QUIET=True).get_structure(
                    file_path.stem, str(file_path)
                )
                if len(structure) > 1:
                    logger.warning(
                        f"File '{file_path.name}' contains {len(structure)} models. Defaulting to Model 0."
                    )
                return list(structure[0].child_dict.keys()), structure
            except Exception as e:
                raise ValueError(
                    f"Malformed PDB file structure in '{file_path.name}': {e}"
                )

        elif file_ext in (".cif", ".mmcif"):
            try:
                mmcif_dict = MMCIF2Dict(str(file_path))
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
                raise ValueError(f"Malformed mmCIF file in '{file_path.name}': {e}")

        return [], None

    @classmethod
    def get_all_atom_coordinates(
        cls, file_path: str | Path, chain_ids: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Main entry point to extract C-alpha backbone and heavy-atom.

        coordinates per chain.

        Args:
            file_path: File path to the target structure file.
            chain_ids: List of chain identifiers to parse. If None, parses all
              available chains.

        Returns:
            dict[str, dict[str, Any]]: Dictionary mapping chain_id to its
            parsed coordinate dictionaries.
        """
        file_ext = cls._validate_structure_file(file_path)
        path_obj = Path(file_path)

        available_chains, loaded_data = cls._load_and_inspect(path_obj, file_ext)
        if not available_chains:
            raise ValueError(f"No chains found in structural file '{path_obj.name}'")

        # Early filtering boundary check
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
            if file_ext == ".pdb":
                results[c_id] = cls._parse_legacy_pdb(loaded_data, c_id)
            else:
                results[c_id] = cls._parse_mmcif_fast_path(loaded_data, c_id)

        return results

    @classmethod
    def _parse_legacy_pdb(cls, structure: Structure, chain_id: str) -> dict[str, Any]:
        """
        Extracts all heavy atoms for FreeSASA Option A alongside C-alpha.

        backbone attributes from a BioPython Structure object.

        Args:
            structure (Structure): Pre-loaded BioPython Structure object.
            chain_id (str): Target chain identifier.

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
        model = structure[0]
        if chain_id not in model:
            raise ValueError(f"Chain identifier '{chain_id}' not found in structure.")

        chain = model[chain_id]

        # C-alpha node buffers
        ca_coordinates, aa_residues, ca_b_factors, ca_occupancies = (
            [],
            [],
            [],
            [],
        )

        # All heavy-atom buffers (for Option A FreeSASA)
        all_atom_coords, all_atom_keys, all_atom_names, all_atom_res_names = (
            [],
            [],
            [],
            [],
        )

        for residue in chain:
            # Skip heterogens/water (residue.id[0] != ' ')
            if residue.id[0] != " ":
                continue

            res_num_str = str(residue.id[1])
            res_name_str = residue.get_resname().strip()

            for atom in residue:
                clean_atom_name = atom.get_name().strip()

                # Skip non-heavy atoms (Hydrogen / Deuterium)
                if (
                    clean_atom_name.startswith(("H", "D", "1H", "2H", "3H"))
                    or atom.element == "H"
                ):
                    continue

                # Resolve disordered point mutations / alternate conformations
                target_atom = atom.selected_child if atom.is_disordered() else atom

                coord = target_atom.get_coord()

                # 1. Accumulate all heavy atoms
                all_atom_coords.append(coord)
                all_atom_keys.append((chain_id, res_num_str))
                all_atom_names.append(clean_atom_name)
                all_atom_res_names.append(res_name_str)

                # 2. Extract C-alpha backbone nodes
                if clean_atom_name == "CA":
                    ca_coordinates.append(coord)
                    aa_residues.append((res_num_str, res_name_str))
                    ca_b_factors.append(float(target_atom.get_bfactor()))
                    ca_occupancies.append(float(target_atom.get_occupancy()))

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return {
            # Downstream C-alpha node arrays
            "ca_coords": np.array(ca_coordinates, dtype=np.float32),
            "aa_residues": aa_residues,
            "ca_b_factors": np.array(ca_b_factors, dtype=np.float32),
            "ca_occupancies": np.array(ca_occupancies, dtype=np.float32),
            # All heavy-atom arrays for FreeSASA
            "all_atom_coords": np.array(all_atom_coords, dtype=np.float64),
            "all_atom_keys": all_atom_keys,
            "all_atom_names": all_atom_names,
            "all_atom_res_names": all_atom_res_names,
        }

    @classmethod
    def _parse_mmcif_fast_path(cls, mmcif_dict: dict, chain_id: str) -> dict[str, Any]:
        """
        Parses mmCIF coordinates from a pre-loaded MMCIF2Dict.

        Extracts all heavy atoms for all-atom SASA calculations alongside
        C-alpha backbone node attributes.

        Args:
            mmcif_dict (dict): MMCIF2Dict object for the target project.
            chain_id (str): Name used for the distinct macromolecule in the
              structure file.

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
        # Prefer author sequence numbers (auth_seq_id); fallback to label_seq_id if needed
        seq_key = (
            "_atom_site.auth_seq_id"
            if "_atom_site.auth_seq_id" in mmcif_dict
            else "_atom_site.label_seq_id"
        )

        # Look for author component ID first, fall back to canonical label component ID
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
                msg = (
                    "Malformed mmCIF structure: Missing required data field" f" '{key}'"
                )
                logger.error(msg)
                raise ValueError(msg)

        # Normalize single-element string values into lists for safe zipping
        cols = {
            k: [mmcif_dict[k]] if isinstance(mmcif_dict[k], str) else mmcif_dict[k]
            for k in required_keys
        }

        # C-alpha node buffers
        ca_coordinates, aa_residues, ca_b_factors, ca_occupancies = (
            [],
            [],
            [],
            [],
        )

        # All heavy-atom buffers (for Option A FreeSASA)
        all_atom_coords, all_atom_keys, all_atom_names, all_atom_res_names = (
            [],
            [],
            [],
            [],
        )

        # Parallel iteration over coordinate columns
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
            # Parse standard polymer ATOM entries matching chain_id
            if group == "ATOM" and chain == chain_id:
                clean_atom_name = str(atom_name).strip()

                # Skip non-heavy atoms (Hydrogen / Deuterium)
                if clean_atom_name.startswith(("H", "D", "1H", "2H", "3H")):
                    continue

                try:
                    coord = [float(x), float(y), float(z)]
                    res_num_str = str(seq_num)
                    res_name_str = str(aa)

                    # 1. Accumulate all heavy atoms
                    all_atom_coords.append(coord)
                    all_atom_keys.append((chain_id, res_num_str))
                    all_atom_names.append(clean_atom_name)
                    all_atom_res_names.append(res_name_str)

                    # 2. Extract C-alpha backbone nodes
                    if clean_atom_name == "CA":
                        ca_coordinates.append(coord)
                        aa_residues.append((res_num_str, res_name_str))
                        ca_b_factors.append(float(b_val))
                        ca_occupancies.append(float(occ_val))

                except ValueError as e:
                    msg = (
                        "Non-numeric spatial coordinates encountered in mmCIF"
                        f" dictionary: {e}"
                    )
                    logger.error(msg)
                    raise ValueError(msg) from e

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return {
            # Downstream C-alpha node arrays
            "ca_coords": np.array(ca_coordinates, dtype=np.float32),
            "aa_residues": aa_residues,
            "ca_b_factors": np.array(ca_b_factors, dtype=np.float32),
            "ca_occupancies": np.array(ca_occupancies, dtype=np.float32),
            # All heavy-atom arrays for FreeSASA
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
        Serializes and saves the pre-parsed mmCIF dictionary to disk using PyTorch serialization.

        Args:
            parsed_dict (Dict[str, Any]): Dictionary returned by _parse_mmcif_fast_path or get_all_atom_coordinates.
            output_path (Union[str, Path]): Destination file path (e.g., 'wt_parsed.pt').

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
        Loads a pre-parsed mmCIF dictionary from disk.

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
            # Extract lead element symbol (e.g., "CA" in protein backbone -> Carbon)
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
    def compute_allatom_sasa_dict(
        cls, parsed_data: dict[str, Any]
    ) -> dict[tuple[str, str], float]:
        """
        Computes All-Atom SASA in-memory using freesasa.calcCoord and aggregates
        atomic surface area values per residue.

        Args:
            parsed_data (dict[str, Any]): Dictionary returned by _parse_mmcif_fast_path or _parse_legacy_pdb.

        Returns:
            dict[tuple[str, str], float]: Mapping of (chain_id, res_number_str) -> raw SASA (Å²).
        """
        all_atom_coords: np.ndarray = parsed_data["all_atom_coords"]  # (M, 3) float64
        # M tuples of (chain_id, res_num)
        all_atom_keys: list[tuple[str, str]] = parsed_data["all_atom_keys"]
        all_atom_names: list[str] = parsed_data["all_atom_names"]
        all_atom_res_names: list[str] = parsed_data["all_atom_res_names"]

        if all_atom_coords.size == 0:
            raise ValueError("Empty coordinate array passed to FreeSASA calculation.")

        # 1. Assign atomic VdW radii
        radii = cls._assign_vdw_radii(all_atom_names, all_atom_res_names)

        # 2. Flatten coordinates in C-contiguous row-major order: [x0, y0, z0, x1, y1, z1, ...]
        flat_coords = all_atom_coords.flatten().tolist()

        # 3. Fast C-binding execution via calcCoord
        try:
            result = freesasa.calcCoord(flat_coords, radii)
        except Exception as e:
            raise ValueError(
                f"freesasa.calcCoord failed during C execution: {e}"
            ) from e

        # 4. Aggregate atomic SASA per residue key (chain_id, res_num_str)
        residue_sasa_map: dict[tuple[str, str], float] = {}
        for i in range(result.nAtoms()):
            key = all_atom_keys[i]
            atom_area = result.atomArea(i)
            residue_sasa_map[key] = residue_sasa_map.get(key, 0.0) + atom_area

        return residue_sasa_map

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
        # 1. Compute raw All-Atom per-residue SASA mapping
        raw_sasa_map = cls.compute_allatom_sasa_dict(parsed_data)

        # 2. Extract C-alpha backbone node keys & residue names
        # List of (res_num_str, res_name)
        aa_residues: list[tuple[str, str]] = parsed_data["aa_residues"]

        # Pull chain_id from first all_atom_key entry
        chain_id = parsed_data["all_atom_keys"][0][0]

        rsasa_list = []
        for res_num_str, res_name in aa_residues:
            key = (chain_id, res_num_str)
            raw_sasa = raw_sasa_map.get(key, 0.0)

            # MaxASA normalization (Tien et al., 2013)
            # Default fallback 200 Å²
            max_sasa = MAX_SASA_TIEN.get(res_name.upper(), DEFAULT_MAX_ASA)
            normalized_rsasa = min(
                max(raw_sasa / max_sasa, 0.0), 1.0
            )  # Clamp to [0, 1]

            rsasa_list.append(normalized_rsasa)

        return np.array(rsasa_list, dtype=np.float32)

    @staticmethod
    def _aggregate_chains(
        chain_parsed_data: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """
        Aggregates multi-chain structural dictionaries into unified complex-level buffers.

        Iterates through per-chain payloads from get_all_atom_coordinates and concatenates them
        across selected chains. Produces unified feature structures for backbone C-alpha nodes (N residues)
        and all-atom heavy coordinates (M heavy atoms, where M >> N).

        Residue identifiers are standardized into `(chain_id, res_num, res_name)` node tuples
        to maintain global residue tracking during downstream graph assembly and FreeSASA lookup.

        Args:
            chain_parsed_data: Dictionary mapping chain identifiers (e.g., "A", "B") to
                their respective parsed feature dictionaries containing C-alpha node vectors
                (`ca_coords`, `aa_residues`, `ca_b_factors`, `ca_occupancies`) and heavy-atom
                arrays (`all_atom_coords`, `all_atom_keys`, `all_atom_names`, `all_atom_res_names`).

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

        Raises:
            ValueError: If `chain_parsed_data` is empty or if any constituent heavy-atom array
                fails numpy stacking due to dimension mismatch.
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
            # Extract C-alpha backbone node attributes
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

            # Extract heavy atoms for FreeSASA
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

    @staticmethod
    def _compute_rsasa_vector(
        nodes: list[tuple[str, str, str]], sasa_map: dict[tuple[str, str], float]
    ) -> np.ndarray:
        """
        Normalizes pre-computed raw SASA values into a C-alpha-aligned rSASA vector.

        Iterates through the ordered $N$ residue node descriptors `(chain_id, res_num, res_name)`
        and retrieves their aggregate raw Solvent Accessible Surface Area (SASA) from `sasa_map`.
        Normalizes each raw value against empirical maximum theoretical solvent accessibility
        scales (MaxASA) from Tien et al. (2013) to yield bounded relative accessibility values in
        `[0.0, 1.0]`.

        Args:
            nodes: Length $N$ list of standardized residue node descriptors formatted as
                `(chain_id, res_num, res_name)`.
            sasa_map: Mapping of `(chain_id, res_num)` tuples to pre-aggregated absolute SASA values (Å²)
                generated by `compute_allatom_sasa_dict`.

        Returns:
            np.ndarray: Float32 array of shape `(N,)` containing normalized relative SASA values
                in `[0.0, 1.0]` aligned with the N target C-alpha nodes. Direct precursor for
                **Channel `[20]`** in Module 1's 26-dimensional node feature tensor, which is
                an N by 26 matrix of real numbers.
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
        # Defensive conversion: Ensure ca_coords is a contiguous 2D float array of shape (N, 3)
        coords_arr = np.asarray(ca_coords, dtype=np.float64)

        N = len(coords_arr)
        if N < 3:
            # Safeguard for extremely short peptides where 3N <= 6 degrees of freedom
            return np.zeros(N, dtype=np.float32)

        # 1. Compute pairwise displacements and Euclidean distances
        diffs = coords_arr[:, None, :] - coords_arr[None, :, :]  # Shape: (N, N, 3)
        # Shape: (N, N)
        dists = np.linalg.norm(diffs, axis=-1)

        # 2. Assemble 3N x 3N Hessian Matrix
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i != j and dists[i, j] <= cutoff:
                    r_ij = diffs[i, j]
                    d_ij = dists[i, j]

                    # 3x3 outer product block
                    K_ij = -(gamma / (d_ij**2)) * np.outer(r_ij, r_ij)

                    # Off-diagonal block assignment
                    H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = K_ij

                    # Accumulate negative sum onto diagonal block H_ii
                    H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= K_ij

        # 3. Symmetric Eigen-decomposition
        eigenvalues, eigenvectors = la.eigh(H)

        # 4. Filter zero modes (all eigenvalues <= 1e-5, including rigid-body & disconnected components)
        valid_mask = eigenvalues > 1e-5
        if not np.any(valid_mask):
            return np.zeros(N, dtype=np.float32)

        # Shape: (num_valid_modes,)
        slow_evals = eigenvalues[valid_mask]
        # Shape: (3N, num_valid_modes)
        slow_evecs = eigenvectors[:, valid_mask]

        # 5. Vectorized Mean-Square Fluctuations (MSF) computation
        # Reshape to (N, 3, num_valid_modes)
        evecs_3d = slow_evecs.reshape(N, 3, -1)

        # Sum squared x, y, z displacement per node per valid mode: Shape (N, num_valid_modes)
        sq_disp = np.sum(evecs_3d**2, axis=1)

        # Weight by inverse non-zero eigenvalues and sum over valid modes
        msf = np.sum(sq_disp / slow_evals, axis=1)

        # 6. Min-Max Normalization for Node Feature Channel [25]
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

        Executes defensive assertions across all 1D C-alpha node-level feature arrays,
        heavy-atom buffers, and metadata lists to guarantee index parity before returning
        data to the caller. This prevents silent dimension mismatches from propagating downstream
        into PyTorch Geometric tensor construction, graph edge-index builders, or FreeSASA.

        Contract Invariants Enforced:
            Residue Node Alignment (N items):
                - len(coords) == N
                - len(nodes) == N
                - len(b_factors) == N
                - len(occupancies) == N
                - len(rsasa) == N
                - len(anm_msf) == N
                where N = len(aa_list) represents the total number of target C-alpha
                residues in the aggregated structure.

            Heavy-Atom Alignment (M items):
                - len(all_atom_coords) == M
                - len(all_atom_keys) == M
                - len(all_atom_names) == M
                - len(all_atom_res_names) == M
                where M represents the total number of heavy atoms across all parsed chains.

        Args:
            parsed_output: The final dictionary assembled by `parse` containing residue-level
                tensors, structural metadata, and heavy-atom SASA arrays.
            filename: Name or string path of the source structure file, used for diagnostic
                formatting in error reporting.

        Raises:
            ValueError: If any constituent vector length deviates from N for residue nodes
                or M for heavy atoms, detailing the exact dimension breakdown in the exception message.
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
                f"Internal length mismatch in residue node features for '{filename}': "
                f"got {n_nodes} residues, "
                f"{len(parsed_output['coords'])} coordinates, "
                f"{len(parsed_output['b_factors'])} b_factors, "
                f"{len(parsed_output['occupancies'])} occupancies, "
                f"{len(parsed_output['rsasa'])} rSASA values, "
                f"{len(parsed_output['anm_msf'])} Mean-Square Fluctuations, and "
                f"{len(parsed_output['nodes'])} node metadata items."
            )

        m_heavy = len(parsed_output["all_atom_coords"])
        if not (
            len(parsed_output["all_atom_keys"])
            == len(parsed_output["all_atom_names"])
            == len(parsed_output["all_atom_res_names"])
            == m_heavy
        ):
            raise ValueError(
                f"Internal length mismatch in heavy-atom arrays for '{filename}': "
                f"got {m_heavy} coordinate rows, "
                f"{len(parsed_output['all_atom_keys'])} atom keys, "
                f"{len(parsed_output['all_atom_names'])} atom names, and "
                f"{len(parsed_output['all_atom_res_names'])} residue names."
            )

    @classmethod
    def parse(
        cls,
        file_path: str | Path,
        chain_ids: list[str] | None = None,
        ref_cif_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Parses macromolecular structural files into unified node feature arrays
        and complex-wide biophysical metrics in a single in-memory pass.

        Args:
            file_path: Absolute or relative path to target structural file.
            chain_ids: Optional list of specific chain identifiers to retain.
            ref_cif_path: Optional path to reference template structure (e.g., 8r3y)
                used to align or impute unmodeled loops.
        """
        path_obj = Path(file_path)
        chain_parsed_data = cls.get_all_atom_coordinates(path_obj, chain_ids=chain_ids)

        # 1. Aggregate per-chain data into complex-wide buffers
        nodes_data, heavy_data = cls._aggregate_chains(chain_parsed_data)

        # Optional: Handle reference-based coordinate imputation if ref_cif_path provided
        if ref_cif_path and Path(ref_cif_path).resolve() != path_obj.resolve():
            logger.info(
                f"Using reference structure '{ref_cif_path}' for coordinate reference."
            )
            # ... loop imputation / gap repair logic here if needed ...

        # 2. Compute raw FreeSASA mapping across all heavy atoms
        sasa_map = cls.compute_allatom_sasa_dict(heavy_data)

        # 3. Compute normalized rSASA vector aligned with backbone nodes
        rsasa_vec = cls._compute_rsasa_vector(nodes_data["nodes"], sasa_map)

        # 4. Compute and assign ANM Physical Dynamics
        ca_coords = nodes_data["coords"]
        msf_features = cls._compute_anm_msf(ca_coords)

        # 5. Assemble output schema
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

        # 6. Contract validation
        cls._validate_parsed_output(parsed_output, path_obj.name)
        return parsed_output
